#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.3, 11/10/2025

import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu

from utils.whitening import inv_sqrt_psd, sample_covariance


COV_SHRINKAGE = 0.02
EIG_FLOOR = 1e-8


def is_positive_semi_definite(matrix, tol=1e-8):
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return eigenvalues.min() >= -tol * np.abs(eigenvalues.max())
    except np.linalg.LinAlgError:
        return False


def compute_orthonormal_basis(V):
    """
    Compute orthonormal basis Q_V for span(V) using QR decomposition.
    
    This is used for stable projection onto span(V) and its orthogonal complement.
    
    Parameters:
    V (np.array): Matrix with columns as basis vectors, shape (K, n_subtypes)
    
    Returns:
    Q_V (np.array): Orthonormal basis for span(V), shape (K, rank)
    P (np.array): Projector onto span(V), P = Q_V @ Q_V.T, shape (K, K)
    P_perp (np.array): Projector onto orthogonal complement, P_perp = I - P, shape (K, K)
    """
    K = V.shape[0]
    
    # QR decomposition: V = Q @ R where Q has orthonormal columns
    Q_V, R = np.linalg.qr(V, mode='reduced')
    
    # Projector onto span(V): P = Q_V @ Q_V.T
    P = Q_V @ Q_V.T
    
    # Orthogonal complement projector: P_perp = I - P
    P_perp = np.eye(K) - P
    
    return Q_V, P, P_perp


def compute_orthogonal_complement_projector(V):
    """
    Compute the orthogonal complement projector P_perp = I - P
    where P = V(V^T V)^{-1} V^T is the projector onto span(V).
    
    Parameters:
    V (np.array): Matrix with columns as basis vectors, shape (K, n_subtypes)
    
    Returns:
    P_perp (np.array): Orthogonal complement projector, shape (K, K)
    """
    _, _, P_perp = compute_orthonormal_basis(V)
    return P_perp


def compute_offtarget_basis(ref_vectors, V, n_components=3):
    """
    Compute OffTarget basis from reference residuals in the orthogonal complement of V.
    
    Uses SVD to extract top variance components (pure variance ordering, no geometric constraints).
    This captures maximum variance in the residual space, providing the most flexible basis for
    modeling unexplained variation.
    
    The basis is ordered by variance:
    - RA0: Direction of maximum residual variance in span(V)^⊥
    - RA1: Direction of 2nd-most residual variance in span(V)^⊥
    - RA2: Direction of 3rd-most residual variance in span(V)^⊥
    
    DETERMINISM: This function is fully deterministic relative to the reference dataset.
    - Depends ONLY on ref_vectors (reference data) and V (computed from reference means)
    - SVD is deterministic (up to sign ambiguity, which doesn't affect energy decomposition)
    - No dependence on test data
    - Always returns exactly min(n_components, rank) axes (capped at 3 by default)
    
    Parameters:
    ref_vectors (list of np.array): Reference vectors (centered, not normalized), each shape (K,)
    V (np.array): Subtype basis matrix, shape (K, n_subtypes)
    n_components (int): Maximum number of OffTarget components (default 3, hard cap)
    
    Returns:
    U_off (np.array): OffTarget basis matrix, shape (K, n_off) where n_off <= min(3, n_components)
                      Columns ordered by descending variance: [RA0, RA1, RA2]
    """
    # Compute orthogonal complement projector
    P_perp = compute_orthogonal_complement_projector(V)
    
    # Project all reference vectors into orthogonal space
    residuals = []
    for x in ref_vectors:
        x_float = np.asarray(x, dtype=float)  # Ensure float type
        r = P_perp @ x_float
        residuals.append(r)
    
    # Stack residuals as rows for SVD - ensure float64 dtype
    R = np.array(residuals, dtype=float)  # shape (n_samples, K)
    
    # Compute SVD: R = U @ S @ Vt
    # Right singular vectors (Vt rows) capture directions of maximum variance
    try:
        U_svd, S, Vt = np.linalg.svd(R, full_matrices=False)
    except np.linalg.LinAlgError:
        # SVD failed, return empty basis
        return np.zeros((V.shape[0], 0))
    
    # Hard cap at 3 components
    n_components = min(n_components, 3)
    
    # Select top n_components by variance (singular values in S are already sorted descending)
    n_off = min(n_components, Vt.shape[0], V.shape[0])
    
    if n_off == 0:
        return np.zeros((V.shape[0], 0))
    
    # Take top n_off components (Vt rows are orthonormal in the row-space of R)
    # After re-projecting with P_perp (numerical safety), we re-orthonormalize
    # deterministically to guarantee U_off.T @ U_off = I.
    U_off_list = []
    for i in range(n_off):
        u = Vt[i, :]  # i-th component (descending variance order)
        
        # Ensure it's in orthogonal complement (numerical safety, should already be true)
        u = P_perp @ u
        
        # Normalize (initial)
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            U_off_list.append(u / norm)
    
    if len(U_off_list) == 0:
        # No significant OffTarget components found
        return np.zeros((V.shape[0], 0))
    
    # Deterministic modified Gram-Schmidt to enforce orthonormal columns
    U_ortho = []
    for u in U_off_list:
        v = u.astype(float, copy=True)
        for q in U_ortho:
            v = v - float(np.dot(q, v)) * q
        nrm = float(np.linalg.norm(v))
        if nrm > 1e-10:
            U_ortho.append(v / nrm)

    if len(U_ortho) == 0:
        return np.zeros((V.shape[0], 0))

    return np.column_stack(U_ortho)


def simplex_log_volume(vectors):
    """
    Calculate the LOG-VOLUME of a k-simplex using stable log-determinant of Gram matrix.
    
    For a k-simplex with vertices p_0, ..., p_k in R^m (with k <= m):
        log(V_k) = 0.5 * log(det(G)) - log(k!)
        where G = E.T @ E is the Gram matrix, E = edge_matrix
    
    This is more stable in high dimensions and plays nicely with additive penalties.
    
    Parameters:
        vectors (list of np.array): A list of k+1 vectors, each describing a point in m-dimensional space.
                                    The dimension of the simplex is k = len(vectors) - 1.
    Returns:
        log_volume (float): The natural log of the k-simplex volume. Returns -inf for degenerate simplices.
    """
    points = np.array(vectors)
    k = len(points) - 1  # Dimension of the simplex
    
    if k < 0:
        return -np.inf
    if k == 0:  # A single point has zero volume
        return -np.inf
    
    # Use first point as origin (translation-invariant)
    E = points[1:] - points[0]  # Edge matrix, shape: (k, m)
    
    # Compute Gram matrix G = E @ E.T
    G = E @ E.T
    
    # Ridge regularization for numerical stability
    eps = 1e-12 * np.trace(G) / max(G.shape[0], 1) if np.trace(G) > 0 else 1e-12
    G_reg = G + eps * np.eye(G.shape[0])
    
    try:
        # Use slogdet for numerical stability (avoids overflow/underflow)
        sign, logdetG = np.linalg.slogdet(G_reg)
        
        if sign <= 0:
            # Non-positive determinant indicates degenerate simplex
            return -np.inf
        
        # log(V_k) = 0.5 * log(det(G)) - log(k!)
        log_vol = 0.5 * logdetG - np.log(math.factorial(k))
        
        # Check for extremely small volume (degenerate)
        if log_vol < -50:  # exp(-50) ≈ 2e-22
            return -np.inf
        
        return log_vol
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf


def maximal_simplex_volume(df: pd.DataFrame, hyperparams: dict = None, verbose: bool = True, max_features: int = 0) -> pd.DataFrame:
    """
    Given a reference data frame containing features and values for select anchoring subtypes, return
    a subset of the dataframe which maximizes the simplex volume defined by those anchoring subtypes while
    minimizing the number of features used.
    
    Note: This function operates on ALREADY TRANSFORMED features (e.g., after log1p scaling).
    
        Parameters:
           df: pandas dataframe with 'Subtype' column and transformed feature columns
           max_features: hard cap on number of features selected in greedy addition
                         (0 = unlimited, default). Prevents runaway with zero-penalty hyperparams.
        Returns:
           out_df: the input df, restricted to the most differentiating features
    """
    # Sort feature columns for determinism (alphabetical order)
    feature_cols = sorted([col for col in df.columns if col != 'Subtype'])
    df = df[['Subtype'] + feature_cols].copy()
    
    n_classes = len(df.Subtype.unique())
    n_features = len(df.columns) - 1
    subtypes = df.Subtype.unique()
    # class_mats are based on the original df's feature columns
    class_mats = [df.loc[df['Subtype'] == subtype].drop('Subtype', axis=1).to_numpy() for subtype in subtypes]
    # df_features_only is used for column names etc. in greedy_maximize and objective
    df_features_only = df.drop('Subtype', axis=1)
    
    if verbose:
        print('Running simplex volume maximization (SVM) for feature subsetting . . .')
        print("Total classes (simplex order + 1): " + str(n_classes))
        print("Total initial features (feature space order): " + str(n_features))
    
    # Precompute Keraon-aligned subtype span basis in the full feature space.
    # Critical: inside the objective we recompute the orthonormal basis in the MASKED feature space
    # (do NOT slice a precomputed Q).
    if "Healthy" not in subtypes:
        raise ValueError("SVM feature selection requires a 'Healthy' subtype in the reference")

    healthy_idx = list(subtypes).index("Healthy")
    all_feature_means = [np.mean(class_mats[i], axis=0) for i in range(len(subtypes))]
    healthy_mean = all_feature_means[healthy_idx]
    disease_subtypes = [st for st in subtypes if st != "Healthy"]

    basis_vectors = []
    for st in disease_subtypes:
        i = list(subtypes).index(st)
        v = all_feature_means[i] - healthy_mean
        v = v / (np.linalg.norm(v) + 1e-12)
        basis_vectors.append(v)

    V_full = np.column_stack(basis_vectors) if basis_vectors else np.zeros((n_features, 0))

    # Whitened geometry: deterministic covariance metric (after standardization).
    X_ref = df_features_only.to_numpy(dtype=float)
    Sigma_global = sample_covariance(X_ref)
    
    # Tunable hyperparameters for objective
    # Can be overridden by passing hyperparams dict (used during CV tuning)
    if hyperparams is None:
        LAMBDA_L0 = 0.0          # L0 penalty (feature count)
        SCATTER_POW = 3.0         # Scatter penalty exponent (higher = tighter classes)
        MARGIN_ALPHA = 1.0        # Margin emphasis (higher = prioritize min gap)
        EDGE_BETA = 0.1           # Edge smoothness (lower = focus on margins)
        GAMMA_REDUNDANCY = 0.1    # Feature redundancy penalty (0.1–0.3, higher = more diversity)
    else:
        LAMBDA_L0 = hyperparams.get('LAMBDA_L0', 0.03)
        SCATTER_POW = hyperparams.get('SCATTER_POW', 2.0)
        MARGIN_ALPHA = hyperparams.get('MARGIN_ALPHA', 0.6)
        EDGE_BETA = hyperparams.get('EDGE_BETA', 0.4)
        GAMMA_REDUNDANCY = hyperparams.get('GAMMA_REDUNDANCY', 0.2)

    def objective(feature_mask):
        """
        Objective function to be maximized (now in log-space for stability).
        
        Combines:
        - Log-volume of simplex in Keraon's subspace
        - Margin-weighted edge metrics (prioritize smallest gap)
        - Within-class scatter penalty
        - L0 feature count penalty
        - Feature redundancy penalty
        
        Parameters:
            feature_mask (np.array): A binary array indicating which features to include.
        Returns:
            score (float): The objective value (higher is better). Returns 0 for degenerate cases.
        """
        current_selection_mask = feature_mask.astype(bool)
        n_selected = int(np.sum(current_selection_mask))
        
        if n_selected == 0:
            return 0.0
        
        # Mask data and compute per-class means
        masked_mats = [mat[:, current_selection_mask] for mat in class_mats]
        masked_vectors = [np.mean(masked_mat, axis=0) for masked_mat in masked_mats 
                         if masked_mat.size > 0 and masked_mat.shape[0] > 0]
        
        if len(masked_vectors) < 2:
            return 0.0
        
        # Whiten in masked space using global covariance metric
        idx = np.where(current_selection_mask)[0]
        Sigma_m = Sigma_global[np.ix_(idx, idx)]
        W_m = inv_sqrt_psd(Sigma_m, shrinkage=COV_SHRINKAGE, eig_floor=EIG_FLOOR)

        means_array = np.vstack(masked_vectors)  # (n_classes, n_selected)
        means_w = (W_m @ means_array.T).T

        # Recompute orthonormal basis in restricted (masked) feature space
        V_mask = V_full[current_selection_mask, :]  # (n_selected, n_disease)
        V_mask_w = W_m @ V_mask
        if V_mask_w.shape[1] > 0:
            _, P_mask_w, _ = compute_orthonormal_basis(V_mask_w)
            means_proj = means_w @ P_mask_w
        else:
            means_proj = means_w
        
        # Compute LOG-VOLUME (more stable in high-D)
        log_vol = simplex_log_volume(list(means_proj))
        if log_vol == -np.inf:
            return 0.0
        
        # Clip edge lengths early to guard degenerate cases
        eps = 1e-12
        pairwise_dist = pdist(means_proj)
        pairwise_dist = np.maximum(pairwise_dist, eps)  # Clip at eps
        
        if pairwise_dist.size == 0:
            return 0.0
        
        # MARGIN-WEIGHTED edge metrics (prioritize min gap for cleaner separation)
        min_edge = np.min(pairwise_dist)
        harmonic_mean_edge = len(pairwise_dist) / np.sum(1.0 / pairwise_dist)
        
        # Combine margin and global smoothness
        log_edge_term = MARGIN_ALPHA * np.log(min_edge + eps) + EDGE_BETA * np.log(harmonic_mean_edge + eps)
        
        # WITHIN-CLASS SCATTER (in whitened geometry)
        masked_cov_mats = []
        for masked_mat_for_cov in masked_mats:
            cov_mat = np.cov(masked_mat_for_cov, rowvar=False)
            
            # Ridge regularization for stability
            if cov_mat.ndim == 2 and cov_mat.shape[0] > 0:
                trace_cov = np.trace(cov_mat)
                K = cov_mat.shape[0]
                ridge_lambda = 1e-6 * trace_cov / K if K > 0 and trace_cov > 0 else 1e-9
                cov_mat = cov_mat + ridge_lambda * np.eye(K)
            elif cov_mat.ndim == 0:
                cov_mat = np.atleast_1d(cov_mat) + 1e-9
            
            if cov_mat.ndim == 2 and cov_mat.shape[0] == W_m.shape[0]:
                cov_mat = W_m @ cov_mat @ W_m.T

            masked_cov_mats.append(cov_mat)
        
        scatter_values = []
        for cov_mat in masked_cov_mats:
            if cov_mat.ndim == 2 and cov_mat.shape[0] > 0:
                diag_vars = np.maximum(np.diag(cov_mat), 0.0)
                scatter = np.sqrt(np.sum(diag_vars))
                scatter_values.append(scatter)
            elif cov_mat.ndim <= 1:
                scatter = np.sqrt(np.maximum(float(np.atleast_1d(cov_mat).flat[0]), 0.0))
                scatter_values.append(scatter)
        
        scatter_values = np.array(scatter_values)
        scatter_values = np.nan_to_num(scatter_values, nan=np.inf, posinf=np.inf, neginf=np.inf)
        finite_scatter = scatter_values[np.isfinite(scatter_values)]
        
        if finite_scatter.size == 0:
            return 0.0
        
        mean_scatter = np.mean(finite_scatter)
        if mean_scatter < eps:
            return 0.0
        
        # Log-scatter penalty (tunable exponent for tighter/looser classes)
        log_scatter_penalty = -SCATTER_POW * np.log(mean_scatter + eps)
        
        # FEATURE REDUNDANCY penalty (correlation among selected features)
        redundancy_penalty = 0.0
        if n_selected > 1 and GAMMA_REDUNDANCY > 0:
            try:
                Xs = df_features_only.iloc[:, current_selection_mask].to_numpy()
                if Xs.shape[0] > 1 and Xs.shape[1] > 1:
                    C = np.corrcoef(Xs, rowvar=False)
                    # Mean absolute off-diagonal correlation
                    triu_indices = np.triu_indices_from(C, k=1)
                    if len(triu_indices[0]) > 0:
                        redundancy = np.nanmean(np.abs(C[triu_indices]))
                        if np.isfinite(redundancy):
                            redundancy_penalty = -GAMMA_REDUNDANCY * redundancy
            except (ValueError, np.linalg.LinAlgError):
                pass  # Skip if correlation fails
        
        # L0 PENALTY (feature count) - encourages fewer features
        l0_penalty = -LAMBDA_L0 * n_selected
        
        # COMBINE in log-space, then exponentiate
        log_score = log_vol + log_edge_term + log_scatter_penalty + redundancy_penalty + l0_penalty
        
        # Exponentiate for final score (or return log_score directly if preferred)
        score = np.exp(log_score) if log_score > -50 else 0.0
        
        return score

    # Determine the hard feature cap for greedy addition
    _max_feat = int(max_features) if max_features > 0 else max(5 * n_classes, 50)

    def greedy_maximize(fun, n, n_min_param): # n_min_param is n_classes - 1
        best_mask = np.zeros(n, dtype=bool)
        best_value = 0.0 # Will be updated after initial selections
        printed_features = set()

        # df_features_only, subtypes, class_mats are accessed from the enclosing scope

        def print_feature_stats_local(feature_name_to_print):
            if verbose and feature_name_to_print not in printed_features:
                print('-------   ' + feature_name_to_print)
                feature_idx_in_df = df_features_only.columns.get_loc(feature_name_to_print)

                for subtype_name, mc_df_mat in zip(subtypes, class_mats):
                    mean_val = np.mean(mc_df_mat[:, feature_idx_in_df])
                    std_val = np.std(mc_df_mat[:, feature_idx_in_df])
                    print(f"{subtype_name}: Mean = {mean_val:.4f}, Std Dev = {std_val:.4f}")
                printed_features.add(feature_name_to_print)

        # --- Step 1: Initial Feature Set Construction ---
        if verbose:
            print('\nStep 1: Initial feature selection for each non-Healthy subtype...')
        healthy_idx = subtypes.tolist().index("Healthy")

        for i_class_idx, subtype_name in enumerate(subtypes):
            if i_class_idx == healthy_idx:
                continue

            current_subtype_samples = class_mats[i_class_idx]
            
            other_subtypes_data_for_mwu = [[] for _ in range(n)]
            for j_other_class_idx, other_subtype_name_iter in enumerate(subtypes):
                if j_other_class_idx == i_class_idx: 
                    continue
                other_subtype_samples_iter = class_mats[j_other_class_idx]
                for k_feature_idx in range(n):
                    if k_feature_idx < other_subtype_samples_iter.shape[1]:
                        other_subtypes_data_for_mwu[k_feature_idx].extend(other_subtype_samples_iter[:, k_feature_idx])

            best_score_for_this_class = -np.inf
            best_feature_for_this_class = -1

            for k_feature_idx in range(n):
                current_subtype_feature_values = current_subtype_samples[:, k_feature_idx]
                other_subtypes_feature_values_for_mwu = np.array(other_subtypes_data_for_mwu[k_feature_idx])

                if len(current_subtype_feature_values) == 0 or len(other_subtypes_feature_values_for_mwu) == 0:
                    continue
                
                # Compute EFFECT SIZE (standardized mean difference) for better seeding
                mean_current = np.mean(current_subtype_feature_values)
                mean_others = np.mean(other_subtypes_feature_values_for_mwu)
                std_current = np.std(current_subtype_feature_values) + 1e-12
                std_others = np.std(other_subtypes_feature_values_for_mwu) + 1e-12
                
                # Cohen's d (pooled standard deviation)
                n1, n2 = len(current_subtype_feature_values), len(other_subtypes_feature_values_for_mwu)
                pooled_std = np.sqrt(((n1 - 1) * std_current**2 + (n2 - 1) * std_others**2) / (n1 + n2 - 2))
                cohens_d = abs(mean_current - mean_others) / (pooled_std + 1e-12)
                
                # Compute separation (absolute mean difference normalized)
                separation = abs(mean_current - mean_others) / (std_current + std_others + 1e-12)
                
                # Mann-Whitney U test (for tie-breaking)
                try:
                    if len(np.unique(current_subtype_feature_values)) < 2 and \
                       len(np.unique(other_subtypes_feature_values_for_mwu)) < 2 and \
                       np.all(current_subtype_feature_values == other_subtypes_feature_values_for_mwu):
                        mwu_p_value = 1.0
                    elif len(np.unique(np.concatenate((current_subtype_feature_values, other_subtypes_feature_values_for_mwu)))) < 2:
                        mwu_p_value = 1.0
                    else:
                        _, mwu_p_value = mannwhitneyu(current_subtype_feature_values, other_subtypes_feature_values_for_mwu, alternative='two-sided', use_continuity=True)
                except ValueError: 
                    mwu_p_value = 1.0
                
                # Combined score: effect size × separation, with MWU as penalty
                # Higher effect size and separation = better feature
                feature_score = cohens_d * separation * (1.0 / (mwu_p_value + 1e-12))

                if feature_score > best_score_for_this_class:
                    best_score_for_this_class = feature_score
                    best_feature_for_this_class = k_feature_idx
            
            if best_feature_for_this_class != -1:
                if not best_mask[best_feature_for_this_class]:
                    best_mask[best_feature_for_this_class] = True
                    feature_name = df_features_only.columns[best_feature_for_this_class]
                    if verbose:
                        print(f"  Selected for '{subtype_name}': '{feature_name}' (effect-size score: {best_score_for_this_class:.4e})")
                    print_feature_stats_local(feature_name)
                else:
                    feature_name = df_features_only.columns[best_feature_for_this_class]
                    if verbose:
                        print(f"  Feature '{feature_name}' (best for '{subtype_name}', score={best_score_for_this_class:.4e}) was already selected for another subtype.")
            else:
                if verbose:
                    print(f"  Could not find a distinct separating feature for '{subtype_name}'.")

        best_value = fun(best_mask)
        if verbose:
            print(f"Initial set of {sum(best_mask)} features selected. Objective score: {best_value:.4e}")

        # --- Step 1.5: Prune to exactly k features (if we have more) ---
        k = n_min_param  # n_classes - 1
        n_initial = sum(best_mask)
        if n_initial > k:
            if verbose:
                print(f"\nStep 1.5: Pruning from {n_initial} to {k} features (minimal simplex dimension)...")
            while sum(best_mask) > k:
                # Find feature whose removal hurts objective least
                selected_indices = np.where(best_mask)[0]
                min_loss = np.inf
                feature_to_remove = -1
                
                for idx in selected_indices:
                    temp_mask = best_mask.copy()
                    temp_mask[idx] = False
                    temp_value = fun(temp_mask)
                    loss = best_value - temp_value
                    
                    if loss < min_loss:
                        min_loss = loss
                        feature_to_remove = idx
                
                if feature_to_remove != -1:
                    best_mask[feature_to_remove] = False
                    best_value = fun(best_mask)
                    removed_name = df_features_only.columns[feature_to_remove]
                    if verbose:
                        print(f"  Removed '{removed_name}' (loss: {min_loss:.4e}, new score: {best_value:.4e})")
                else:
                    break  # Safety exit
            
            if verbose:
                print(f"Pruned to {sum(best_mask)} features. Lean core established.")

        # --- Step 2: Refinement Loop (Iterative Replacement Scan) ---
        if verbose:
            print('\nStep 2: Refining by attempting feature replacements (iterative)...')
        
        while True: # Outer loop for iterative refinement
            improvement_made_in_this_pass = False
            # Get the current set of selected features for this pass
            indices_to_consider_for_replacement = np.where(best_mask)[0].tolist()
            
            if not indices_to_consider_for_replacement:
                if verbose:
                    print("  No features currently selected to refine.")
                break

            for idx_to_replace in indices_to_consider_for_replacement:
                # Check if the feature to replace is still in the best_mask,
                # as it might have been replaced by a previous iteration in *this same pass*
                # if it was also a candidate for replacement earlier in the list.
                # Or, more simply, ensure it's part of the current best_mask before trying to improve it.
                if not best_mask[idx_to_replace]:
                    continue

                objective_at_start_of_replacing_this_idx = fun(best_mask) # Objective before trying to replace idx_to_replace
                
                temp_mask_for_replacement = best_mask.copy()
                temp_mask_for_replacement[idx_to_replace] = False # Temporarily remove
                
                best_replacement_candidate_idx = -1
                # Initialize with a value that any actual objective should beat if it's an improvement
                current_best_objective_with_a_replacement = objective_at_start_of_replacing_this_idx 
                
                found_potential_replacement_for_this_idx = False

                for i_candidate_replacement in range(n):
                    # Skip if it's the one we just removed or if it's already in the set (excluding the removed one)
                    if i_candidate_replacement == idx_to_replace or temp_mask_for_replacement[i_candidate_replacement]:
                        continue

                    candidate_mask_with_replacement = temp_mask_for_replacement.copy()
                    candidate_mask_with_replacement[i_candidate_replacement] = True
                    
                    objective_of_this_specific_replacement = fun(candidate_mask_with_replacement)

                    # We are looking for a replacement that improves upon the current best_objective_with_a_replacement
                    # AND is better than the objective before we even started trying to replace idx_to_replace
                    if objective_of_this_specific_replacement > current_best_objective_with_a_replacement:
                        current_best_objective_with_a_replacement = objective_of_this_specific_replacement
                        best_replacement_candidate_idx = i_candidate_replacement
                        found_potential_replacement_for_this_idx = True
                
                # If a better replacement was found for idx_to_replace AND it improves the overall best_value
                if found_potential_replacement_for_this_idx and current_best_objective_with_a_replacement > objective_at_start_of_replacing_this_idx:
                    old_feature_name = df_features_only.columns[idx_to_replace]
                    new_feature_name = df_features_only.columns[best_replacement_candidate_idx]
                    
                    best_mask[idx_to_replace] = False
                    best_mask[best_replacement_candidate_idx] = True
                    
                    if verbose:
                        print(f"  Replaced '{old_feature_name}' with '{new_feature_name}'.")
                        print(f"  Scaled simplex volume: {objective_at_start_of_replacing_this_idx:.4e} -> {current_best_objective_with_a_replacement:.4e}")
                    
                    best_value = current_best_objective_with_a_replacement # Update global best_value
                    print_feature_stats_local(new_feature_name)
                    improvement_made_in_this_pass = True
            
            if not improvement_made_in_this_pass:
                if verbose:
                    print("  No further improvements found in this refinement pass.")
                break # Exit the outer while loop

        if verbose:
            print(f"After refinement, {sum(best_mask)} features. Scaled simplex volume: {best_value:.4e}")

        # --- Step 3: Iterative Greedy Addition ---
        if verbose:
            print('\nStep 3: Continuing greedy maximization . . .\n')
        while True:
            num_selected_features = sum(best_mask)
            
            if num_selected_features >= n:
                if verbose:
                    print("All features have been selected.")
                break

            if num_selected_features >= _max_feat:
                if verbose:
                    print(f"Reached max_features cap ({_max_feat}). Stopping greedy addition.")
                break

            overall_best_next_value_this_pass = -np.inf 
            feature_to_add_this_pass = -1 

            for i_candidate_feature in range(n):
                if not best_mask[i_candidate_feature]: 
                    temp_mask = best_mask.copy()
                    temp_mask[i_candidate_feature] = True
                    current_eval_value = fun(temp_mask)
                    
                    if current_eval_value > overall_best_next_value_this_pass:
                        overall_best_next_value_this_pass = current_eval_value
                        feature_to_add_this_pass = i_candidate_feature
            
            if feature_to_add_this_pass == -1:
                if verbose:
                    print("Stopping greedy addition: No feature could be found that yields a better (or finite) objective value in this pass.")
                break

            perform_addition = False
            if num_selected_features < n_min_param:
                if overall_best_next_value_this_pass > best_value:
                     perform_addition = True
                else: # No improvement found even when trying to reach n_min_param
                    if verbose:
                        print(f"Stopping: No feature improves objective ({best_value:.4e}), even while trying to reach n_min_param ({n_min_param}). Selected {num_selected_features}.")
                    break
            else: # At or above n_min_param, require a small relative improvement to keep adding features
                if overall_best_next_value_this_pass > best_value: 
                    if best_value <= 1e-9: 
                        if overall_best_next_value_this_pass > best_value + 1e-9: 
                            perform_addition = True
                    elif (overall_best_next_value_this_pass - best_value) / abs(best_value) > 0.001: 
                        perform_addition = True
                
                if not perform_addition:
                    if verbose:
                        print(f"Stopping addition (post n_min_param): Best potential improvement to {overall_best_next_value_this_pass:.4e} (from {best_value:.4e}) does not meet criteria or is not an improvement.")
                    break
            
            if perform_addition:
                added_feature_name = df_features_only.columns[feature_to_add_this_pass]
                
                if verbose:
                    print(f"Adding feature '{added_feature_name}'. Scaled simplex volume: {best_value:.4e} -> {overall_best_next_value_this_pass:.4e}")
                best_mask[feature_to_add_this_pass] = True
                best_value = overall_best_next_value_this_pass 
                print_feature_stats_local(added_feature_name)
            else:
                if num_selected_features < n_min_param: # This case should be caught by the break above if no improvement
                    if verbose:
                        print(f"Stopping: No improving feature found to add. Currently {num_selected_features} features, n_min_param is {n_min_param}.")
                break 
                
        return best_mask

    result_mask = greedy_maximize(objective, n_features, n_classes - 1)
    final_objective_value = objective(result_mask)
    if verbose:
        print('\nMaximal simplex volume feature subsetting complete.')
        print('Final scaled simplex volume: ' + str(final_objective_value))
        print('Final number of features: ' + str(sum(result_mask)))
    return df_features_only[df_features_only.columns[result_mask.astype(bool)]]


def svm_objective_for_features(df: pd.DataFrame, selected_features: list[str], hyperparams: dict | None = None) -> float:
    """Compute the SVM objective value for a fixed set of features.

    This is used for stability-selection tie-breaking on the full reference dataset.
    """
    if "Subtype" not in df.columns:
        raise ValueError("df must include 'Subtype'")

    if len(selected_features) == 0:
        return 0.0

    df_sel = df[["Subtype"] + list(selected_features)].copy()
    # Use maximal_simplex_volume internals by evaluating its objective via a single-run greedy mask.
    # To avoid duplicating all internal code, we compute the score by running a single evaluation
    # of the nested objective (replicated here in compact form).

    feature_cols = sorted([c for c in df_sel.columns if c != "Subtype"])
    df_sel = df_sel[["Subtype"] + feature_cols].copy()

    subtypes = df_sel.Subtype.unique()
    class_mats = [df_sel.loc[df_sel["Subtype"] == st].drop("Subtype", axis=1).to_numpy(dtype=float) for st in subtypes]
    df_features_only = df_sel.drop("Subtype", axis=1)

    if hyperparams is None:
        LAMBDA_L0 = 0.0
        SCATTER_POW = 3.0
        MARGIN_ALPHA = 1.0
        EDGE_BETA = 0.1
        GAMMA_REDUNDANCY = 0.1
    else:
        LAMBDA_L0 = hyperparams.get("LAMBDA_L0", 0.03)
        SCATTER_POW = hyperparams.get("SCATTER_POW", 2.0)
        MARGIN_ALPHA = hyperparams.get("MARGIN_ALPHA", 0.6)
        EDGE_BETA = hyperparams.get("EDGE_BETA", 0.4)
        GAMMA_REDUNDANCY = hyperparams.get("GAMMA_REDUNDANCY", 0.2)

    if "Healthy" not in subtypes:
        return 0.0
    healthy_idx = list(subtypes).index("Healthy")
    all_feature_means = [np.mean(class_mats[i], axis=0) for i in range(len(subtypes))]
    healthy_mean = all_feature_means[healthy_idx]
    disease_subtypes = [st for st in subtypes if st != "Healthy"]
    basis_vectors = []
    for st in disease_subtypes:
        i = list(subtypes).index(st)
        v = all_feature_means[i] - healthy_mean
        v = v / (np.linalg.norm(v) + 1e-12)
        basis_vectors.append(v)
    V_full = np.column_stack(basis_vectors) if basis_vectors else np.zeros((len(feature_cols), 0))

    X_ref = df_features_only.to_numpy(dtype=float)
    Sigma_global = sample_covariance(X_ref)

    feature_mask = np.ones(len(feature_cols), dtype=bool)
    n_selected = len(feature_cols)
    masked_mats = [mat[:, feature_mask] for mat in class_mats]
    masked_vectors = [np.mean(m, axis=0) for m in masked_mats if m.size > 0 and m.shape[0] > 0]
    if len(masked_vectors) < 2:
        return 0.0

    Sigma_m = Sigma_global
    W_m = inv_sqrt_psd(Sigma_m, shrinkage=COV_SHRINKAGE, eig_floor=EIG_FLOOR)
    means_array = np.vstack(masked_vectors)
    means_w = (W_m @ means_array.T).T
    V_mask_w = W_m @ V_full
    if V_mask_w.shape[1] > 0:
        _, P_mask_w, _ = compute_orthonormal_basis(V_mask_w)
        means_proj = means_w @ P_mask_w
    else:
        means_proj = means_w

    log_vol = simplex_log_volume(list(means_proj))
    if log_vol == -np.inf:
        return 0.0

    eps = 1e-12
    pairwise_dist = np.maximum(pdist(means_proj), eps)
    if pairwise_dist.size == 0:
        return 0.0
    min_edge = float(np.min(pairwise_dist))
    harmonic_mean_edge = float(len(pairwise_dist) / np.sum(1.0 / pairwise_dist))
    log_edge_term = MARGIN_ALPHA * np.log(min_edge + eps) + EDGE_BETA * np.log(harmonic_mean_edge + eps)

    scatter_vals = []
    for m in masked_mats:
        cov = np.cov(m, rowvar=False)
        if cov.ndim == 2 and cov.shape[0] > 0:
            tr = np.trace(cov)
            K = cov.shape[0]
            cov = cov + (1e-6 * tr / K if tr > 0 else 1e-9) * np.eye(K)
            cov = W_m @ cov @ W_m.T
            scatter_vals.append(float(np.sqrt(np.sum(np.maximum(np.diag(cov), 0.0)))))
        else:
            scatter_vals.append(0.0)
    mean_scatter = float(np.mean([v for v in scatter_vals if np.isfinite(v)]))
    if mean_scatter < eps:
        return 0.0
    log_scatter_penalty = -SCATTER_POW * np.log(mean_scatter + eps)

    redundancy_penalty = 0.0
    if n_selected > 1 and GAMMA_REDUNDANCY > 0:
        try:
            C = np.corrcoef(X_ref, rowvar=False)
            triu = np.triu_indices_from(C, k=1)
            if len(triu[0]) > 0:
                redundancy = np.nanmean(np.abs(C[triu]))
                if np.isfinite(redundancy):
                    redundancy_penalty = -GAMMA_REDUNDANCY * float(redundancy)
        except (ValueError, np.linalg.LinAlgError):
            pass

    l0_penalty = -LAMBDA_L0 * n_selected
    log_score = log_vol + log_edge_term + log_scatter_penalty + redundancy_penalty + l0_penalty
    return float(np.exp(log_score) if log_score > -50 else 0.0)


def _run_single_bootstrap(
    ref_df: pd.DataFrame,
    feature_cols: list,
    idx_by_subtype: dict,
    subsample: float,
    params: dict,
    rng_seed: int,
    max_features: int,
) -> set:
    """Run a single bootstrap iteration for stability selection (picklable top-level function)."""
    rng = np.random.default_rng(rng_seed)
    boot_idx = []
    for _st, idxs in sorted(idx_by_subtype.items()):
        if len(idxs) == 0:
            continue
        n_take = max(1, int(round(subsample * len(idxs))))
        boot_idx.extend(rng.choice(idxs, size=n_take, replace=True).tolist())
    boot_df = ref_df.loc[boot_idx, ["Subtype"] + feature_cols].copy()
    boot_selected = maximal_simplex_volume(boot_df, hyperparams=params, verbose=False, max_features=max_features)
    return set(boot_selected.columns)


def stability_select_svm_hyperparams(
    ref_df: pd.DataFrame,
    param_grid: dict,
    n_boot: int = 200,
    subsample: float = 0.8,
    seed: int = 23,
    freq_threshold: float = 0.7,
    min_features: int = 0,
    n_jobs: int = -1,
    verbose: bool = True,
):
    """Reference-only stability selection to freeze SVM hyperparameters and stable feature set."""
    import sys
    from itertools import product
    from joblib import Parallel, delayed

    if "Subtype" not in ref_df.columns:
        raise ValueError("ref_df must contain 'Subtype'")

    rng = np.random.default_rng(seed)
    feature_cols = sorted([c for c in ref_df.columns if c != "Subtype"])
    y = ref_df["Subtype"].astype(str)
    n_classes = len(y.unique())
    # Hard cap for greedy addition inside each bootstrap (prevents runaway with zero-penalty combos)
    max_features_cap = max(5 * n_classes, 50)

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = [dict(zip(param_names, v)) for v in product(*param_values)]
    if len(combos) == 0:
        raise ValueError("param_grid produced zero combinations")

    idx_by_subtype = {st: ref_df.index[y == st].to_numpy() for st in sorted(y.unique())}

    best_params = None
    best_freq_table = None
    best_stable = None
    best_score = (-np.inf, -np.inf, np.inf)  # (stability, full_objective, size)

    # Pre-generate independent seeds for reproducibility across combos/boots
    all_boot_seeds = rng.integers(0, 2**31, size=(len(combos), n_boot)).tolist()

    for combo_i, params in enumerate(combos):
        if verbose:
            print(f"Stability selection: evaluating {combo_i + 1}/{len(combos)}: {params}")
            sys.stdout.flush()

        boot_seeds = all_boot_seeds[combo_i]
        selected_sets: list[set[str]] = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_bootstrap)(
                ref_df, feature_cols, idx_by_subtype, subsample, params, boot_seeds[b], max_features_cap
            )
            for b in range(n_boot)
        )

        # Frequencies
        counts = {f: 0 for f in feature_cols}
        for s in selected_sets:
            for f in s:
                counts[f] += 1
        freqs = {f: counts[f] / n_boot for f in feature_cols}

        freq_table = pd.DataFrame({"feature": list(freqs.keys()), "frequency": list(freqs.values())}).sort_values(
            ["frequency", "feature"], ascending=[False, True]
        )

        # Stability: mean pairwise Jaccard
        if len(selected_sets) > 1:
            sets_list = selected_sets
            jacc = []
            for i in range(len(sets_list)):
                for j in range(i + 1, len(sets_list)):
                    a = sets_list[i]
                    b = sets_list[j]
                    denom = len(a | b)
                    jacc.append((len(a & b) / denom) if denom > 0 else 1.0)
            stability = float(np.mean(jacc)) if jacc else 0.0
        else:
            stability = 0.0

        # Stable features are those that appear in > freq_threshold of bootstraps.
        # (Strict inequality matches the intended interpretation of a cutoff like "> 0.10".)
        stable = freq_table.loc[freq_table["frequency"] > freq_threshold, "feature"].tolist()
        if len(stable) == 0:
            sizes = sorted([len(s) for s in selected_sets])
            k = int(np.median(sizes)) if sizes else 0
            stable = freq_table.head(k)["feature"].tolist()

        # Enforce a minimum feature count by topping up from the highest-frequency features.
        if int(min_features) > 0 and len(stable) < int(min_features):
            need = int(min_features) - len(stable)
            extras = [f for f in freq_table["feature"].tolist() if f not in stable]
            stable = stable + extras[:need]

        full_obj = svm_objective_for_features(ref_df, stable, hyperparams=params) if len(stable) > 0 else 0.0
        score = (stability, full_obj, len(stable))

        if score[0] > best_score[0] or (score[0] == best_score[0] and (score[1] > best_score[1] or (score[1] == best_score[1] and score[2] < best_score[2]))):
            best_score = score
            best_params = params
            best_freq_table = freq_table
            best_stable = stable

    return best_params, best_freq_table, best_stable
