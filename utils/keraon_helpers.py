#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v2.2, 5/2/2024

import math
import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.special import softmax
from scipy.spatial.distance import pdist
from scipy.stats import multivariate_normal, mannwhitneyu


def is_positive_semi_definite(matrix, tol=1e-8):
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return eigenvalues.min() >= -tol * np.abs(eigenvalues.max())
    except np.linalg.LinAlgError:
        return False

def calculate_log_likelihoods(tfx, feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes):
    """
    Calculate log likelihoods for given TFX value and feature values.
    """
    log_likelihoods = []
    for subtype in subtypes:
        mu_mixture = tfx * mu_subs[subtypes.index(subtype)] + (1 - tfx) * mu_healthy
        cov_mixture = np.eye(cov_healthy.shape[0])
        log_likelihood = multivariate_normal.logpdf(feature_vals, mean=mu_mixture, cov=cov_mixture)
        log_likelihoods.append(log_likelihood)
    return log_likelihoods

def optimize_tfx(feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes):
    """
    Optimize 'TFX' to maximize total log likelihood.
    """
    best_tfx = 0
    best_ll = float('inf')

    for try_tfx in np.arange(0, 1.001, 0.001):
        log_likelihoods = calculate_log_likelihoods(try_tfx, feature_vals, mu_healthy, cov_healthy, mu_subs, subtypes)
        ll = -np.sum(np.exp(log_likelihoods))
        if ll < best_ll:
            best_ll = ll
            best_tfx = try_tfx

    return best_tfx

def update_predictions(predictions, sample, tfx, tfx_shifted, log_likelihoods, subtypes):
    """
    Update predictions DataFrame with calculated values.
    """
    weights = softmax(log_likelihoods)

    predictions.loc[sample, 'TFX'] = tfx
    predictions.loc[sample, 'TFX_shifted'] = tfx_shifted
    max_weight = 0
    max_subtype = 'NoSolution'
    for subtype in subtypes:
        weight = np.round(weights[subtypes.index(subtype)], 4)
        predictions.loc[sample, subtype] = weight
        if weight > max_weight:
            max_weight = weight
            max_subtype = subtype

    predictions.loc[sample, 'Prediction'] = max_subtype

def gram_schmidt(vectors):
    """
    Perform the Gram-Schmidt process to orthogonalize a set of vectors.
    """
    basis = []
    for v in vectors:
        w = v - sum(np.dot(v,b)*b for b in basis)
        basis.append(w/np.linalg.norm(w))
    return np.array(basis)

def transform_to_basis(vector, basis):
    """
    Project a vector onto a new basis and calculate the difference between the original vector and its projection.
    """
    # Project the original vector onto the new basis
    projected_vector = np.dot(vector, basis.T)
    # Project the transformed vector back into the original space
    back_projected_vector = np.dot(projected_vector, basis)
    # Calculate the difference between the original vector and its projection
    difference_vector = vector - back_projected_vector
    return projected_vector, difference_vector


def simplex_volume(vectors):
    """
    Calculate the volume of a simplex in n-dimensional space using the Cayley-Menger determinant.
    Parameters:
        vectors (list of np.array): A list of n+1 vectors, each describing a point in n-dimensional space.
                                    The dimension of the space is m (len(vectors[0])).
                                    The dimension of the simplex is k = len(vectors) - 1.
    Returns:
        volume (float): The volume of the simplex.
    """
    points = np.array(vectors)
    k = len(points) - 1  # Dimension of the simplex
    if k < 0:
        return 0.0
    if k == 0: # A single point
        return 0.0
    
    # Calculate squared distances between points
    dist_sq = np.zeros((k + 1, k + 1))
    for i in range(k + 1):
        for j in range(i + 1, k + 1):
            d = np.linalg.norm(points[i] - points[j])
            dist_sq[i, j] = d**2
            dist_sq[j, i] = d**2

    # Construct the Cayley-Menger matrix
    cm_matrix = np.zeros((k + 2, k + 2))
    cm_matrix[0, 0] = 0
    cm_matrix[0, 1:] = 1
    cm_matrix[1:, 0] = 1
    cm_matrix[1:, 1:] = dist_sq
    # Calculate the determinant
    with np.errstate(invalid='ignore'):
        det_cm = np.linalg.det(cm_matrix)
    if np.isnan(det_cm):
        return 0.0
    # Cayley-Menger determinant formula for k-simplex volume V_k:
    # V_k^2 = (-1)^(k+1) / (2^k * (k!)^2) * det(CM_matrix)
    numerator = (-1)**(k + 1) * det_cm
    denominator = (2**k) * (math.factorial(k)**2)
    vol_sq = numerator / denominator

    if vol_sq < 1e-9: # If squared volume is very close to zero or negative (due to precision)
        return 0.0
    
    return np.sqrt(vol_sq)


def maximal_simplex_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a reference data frame containing features and values for select anchoring subtypes, return
    a subset of the dataframe which maximizes the simplex volume defined by those anchoring subtypes while
    minimizing the number of features used.
        Parameters:
           df: pandas dataframe
        Returns:
           out_df: the input df, restricted to the most differentiating features
    """
    n_classes = len(df.Subtype.unique())
    n_features = len(df.columns) - 1
    subtypes = df.Subtype.unique()
    # class_mats are based on the original df's feature columns
    class_mats = [df.loc[df['Subtype'] == subtype].drop('Subtype', axis=1).to_numpy() for subtype in subtypes]
    # df_features_only is used for column names etc. in greedy_maximize and objective
    df_features_only = df.drop('Subtype', axis=1)
    
    print('Running simplex volumne maximization (SVM) for feature subsetting . . .')
    print("Total classes (simplex order + 1): " + str(n_classes))
    print("Total initial features (feature space order): " + str(n_features))

    def objective(feature_mask):
        """
        Objective function to be maximized.
        Parameters:
            feature_mask (np.array): A binary array indicating which features to include.
        Returns:
            volume (float): The volume of the simplex formed by the vectors, scaled.
        """
        current_selection_mask = feature_mask.astype(bool)
        masked_mats = [mat[:, current_selection_mask] for mat in class_mats]
        masked_vectors = [np.mean(masked_mat, axis=0) for masked_mat in masked_mats if masked_mat.size > 0 and masked_mat.shape[0] > 0]
        
        volume = simplex_volume(masked_vectors)
        # If volume is essentially zero, no need to calculate scale factor.
        if volume < 1e-9: # Consistent with simplex_volume's internal threshold
            return 0.0

        masked_cov_mats = []
        for masked_mat_for_cov in masked_mats:
            cov_mat = np.cov(masked_mat_for_cov, rowvar=False)
            masked_cov_mats.append(cov_mat)

        psd_stati = [is_positive_semi_definite(cov) for cov in masked_cov_mats]
        if not all(psd_stati):
            return 0.0

        # Calculate Harmonic Mean of Edge Lengths (bias towards smaller values)
        points_for_pdist = np.array([v.flatten() for v in masked_vectors])
        pairwise_dist = pdist(points_for_pdist) 

        if pairwise_dist.size == 0: 
            harmonic_mean_edge_length = 0.0
            regulatory_term = 0.0
        elif np.any(pairwise_dist < 1e-9): # Check for zero or near-zero distances
            harmonic_mean_edge_length = 0.0
            regulatory_term = 0.0
        else:
            # All distances are > 1e-9, safe for reciprocal
            harmonic_mean_edge_length = len(pairwise_dist) / np.sum(1.0 / pairwise_dist)
            # Compute regulatory term, min_edge_length / max_edge_length (punish irregular shapes)
            min_edge_length = np.min(pairwise_dist)
            max_edge_length = np.max(pairwise_dist)
            regulatory_term = min_edge_length / max_edge_length

        
        class_variances_selected_features = []
        num_selected_features = int(np.sum(current_selection_mask))

        for cov_mat in masked_cov_mats: # These cov_mats passed PSD check
            if cov_mat.ndim == 2 and cov_mat.shape[0] > 0 and cov_mat.shape[0] == cov_mat.shape[1]:
                 diag_vars = np.diag(cov_mat)
                 diag_vars = np.maximum(diag_vars, 0.0) # Ensure non-negativity
                 diag_vars = np.nan_to_num(diag_vars, nan=0.0, posinf=0.0, neginf=0.0) # Clean up
                 class_variances_selected_features.append(diag_vars)
            else:
                 # Fallback for unexpected cov_mat structure after PSD checks
                 class_variances_selected_features.append(np.zeros(num_selected_features if num_selected_features > 0 else 1))
        
        sum_of_variances_per_class = np.array([np.sum(vars_for_class) if len(vars_for_class) > 0 else 0.0
                                               for vars_for_class in class_variances_selected_features])

        sum_of_variances_per_class = np.nan_to_num(sum_of_variances_per_class, nan=np.inf, posinf=np.inf, neginf=np.inf)
        sum_of_variances_per_class[sum_of_variances_per_class < 0] = np.inf
        
        finite_sum_variances = sum_of_variances_per_class[np.isfinite(sum_of_variances_per_class)]

        if finite_sum_variances.size == 0:
            mean_scatter_val = np.inf # All class variances were infinite or no valid classes
        else:
            # finite_sum_variances are already non-negative and finite.
            scatter_values_per_class = np.sqrt(finite_sum_variances) 
            mean_scatter_val = np.mean(scatter_values_per_class)

        # Scale Factor
        if harmonic_mean_edge_length < 1e-9: # If harmonic mean of edges is (near) zero
            scale_factor = 0.0
        elif mean_scatter_val == np.inf or (mean_scatter_val + 1e-9) < 1e-9: # Avoid division by inf or (near) zero
            scale_factor = 0.0
        else:
            scale_factor = harmonic_mean_edge_length / (mean_scatter_val + 1e-9)**(3/2)
    
        return volume * scale_factor * regulatory_term

    def greedy_maximize(fun, n, n_min_param): # n_min_param is n_classes - 1
        best_mask = np.zeros(n, dtype=bool)
        best_value = 0.0 # Will be updated after initial selections
        printed_features = set()

        # df_features_only, subtypes, class_mats are accessed from the enclosing scope

        def print_feature_stats_local(feature_name_to_print):
            if feature_name_to_print not in printed_features:
                print('-------   ' + feature_name_to_print)
                feature_idx_in_df = df_features_only.columns.get_loc(feature_name_to_print)

                for subtype_name, mc_df_mat in zip(subtypes, class_mats):
                    mean_val = np.mean(mc_df_mat[:, feature_idx_in_df])
                    std_val = np.std(mc_df_mat[:, feature_idx_in_df])
                    print(f"{subtype_name}: Mean = {mean_val:.4f}, Std Dev = {std_val:.4f}")
                printed_features.add(feature_name_to_print)

        # --- Step 1: Initial Feature Set Construction ---
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

            best_mwu_p_for_this_class = 1.1
            best_feature_for_this_class = -1

            for k_feature_idx in range(n):
                current_subtype_feature_values = current_subtype_samples[:, k_feature_idx]
                other_subtypes_feature_values_for_mwu = np.array(other_subtypes_data_for_mwu[k_feature_idx])

                if len(current_subtype_feature_values) == 0 or len(other_subtypes_feature_values_for_mwu) == 0:
                    continue
                
                try:
                    # Mann-Whitney U test requires at least one observation in each group and some variance.
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

                if mwu_p_value < best_mwu_p_for_this_class:
                    best_mwu_p_for_this_class = mwu_p_value
                    best_feature_for_this_class = k_feature_idx
            
            if best_feature_for_this_class != -1:
                if not best_mask[best_feature_for_this_class]:
                    best_mask[best_feature_for_this_class] = True
                    feature_name = df_features_only.columns[best_feature_for_this_class]
                    print(f"  Selected for '{subtype_name}': '{feature_name}' (MWU p-value: {best_mwu_p_for_this_class:.4e})")
                    print_feature_stats_local(feature_name)
                else:
                    feature_name = df_features_only.columns[best_feature_for_this_class]
                    print(f"  Feature '{feature_name}' (best for '{subtype_name}', p={best_mwu_p_for_this_class:.4e}) was already selected for another subtype.")
            else:
                print(f"  Could not find a distinct separating feature for '{subtype_name}'.")

        best_value = fun(best_mask)
        print(f"Initial set of {sum(best_mask)} features selected. Scaled simplex volume: {best_value:.4e}")

        # --- Step 2: Refinement Loop (Iterative Replacement Scan) ---
        print('\nStep 2: Refining initial set by attempting feature replacements (iterative)...')
        
        while True: # Outer loop for iterative refinement
            improvement_made_in_this_pass = False
            # Get the current set of selected features for this pass
            indices_to_consider_for_replacement = np.where(best_mask)[0].tolist()
            
            if not indices_to_consider_for_replacement:
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
                    
                    print(f"  Replaced '{old_feature_name}' with '{new_feature_name}'.")
                    print(f"  Scaled simplex volume: {objective_at_start_of_replacing_this_idx:.4e} -> {current_best_objective_with_a_replacement:.4e}")
                    
                    best_value = current_best_objective_with_a_replacement # Update global best_value
                    print_feature_stats_local(new_feature_name)
                    improvement_made_in_this_pass = True
            
            if not improvement_made_in_this_pass:
                print("  No further improvements found in this refinement pass.")
                break # Exit the outer while loop

        print(f"After refinement, {sum(best_mask)} features. Scaled simplex volume: {best_value:.4e}")

        # --- Step 3: Iterative Greedy Addition ---
        print('\nStep 3: Continuing greedy maximization . . .\n')
        while True:
            num_selected_features = sum(best_mask)
            
            if num_selected_features >= n:
                print("All features have been selected.")
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
                print("Stopping greedy addition: No feature could be found that yields a better (or finite) objective value in this pass.")
                break

            perform_addition = False
            if num_selected_features < n_min_param:
                if overall_best_next_value_this_pass > best_value:
                     perform_addition = True
                else: # No improvement found even when trying to reach n_min_param
                    print(f"Stopping: No feature improves objective ({best_value:.4e}), even while trying to reach n_min_param ({n_min_param}). Selected {num_selected_features}.")
                    break
            else: # At or above n_min_param, check for >1% improvement
                if overall_best_next_value_this_pass > best_value: 
                    if best_value <= 1e-9: 
                        if overall_best_next_value_this_pass > best_value + 1e-9: 
                            perform_addition = True
                    elif (overall_best_next_value_this_pass - best_value) / abs(best_value) > 0.01: 
                        perform_addition = True
                
                if not perform_addition:
                    print(f"Stopping addition (post n_min_param): Best potential improvement to {overall_best_next_value_this_pass:.4e} (from {best_value:.4e}) does not meet >1% criteria or is not an improvement.")
                    break
            
            if perform_addition:
                added_feature_name = df_features_only.columns[feature_to_add_this_pass]
                
                print(f"Adding feature '{added_feature_name}'. Scaled simplex volume: {best_value:.4e} -> {overall_best_next_value_this_pass:.4e}")
                best_mask[feature_to_add_this_pass] = True
                best_value = overall_best_next_value_this_pass 
                print_feature_stats_local(added_feature_name)
            else:
                if num_selected_features < n_min_param: # This case should be caught by the break above if no improvement
                    print(f"Stopping: No improving feature found to add. Currently {num_selected_features} features, n_min_param is {n_min_param}.")
                break 
                
        return best_mask

    result_mask = greedy_maximize(lambda x: objective(x), n_features, n_classes - 1)
    final_objective_value = objective(result_mask)
    print('\nMaximal simplex volume feature subsetting complete.')
    print('Final scaled simplex volume: ' + str(final_objective_value))
    print('Final number of features: ' + str(sum(result_mask)))
    return df_features_only[df_features_only.columns[result_mask.astype(bool)]]


def specificity_sensitivity(target: np.array, predicted: np.array, threshold: float) -> tuple:
    """
    Calculate the specificity and sensitivity for a given threshold.

    Parameters:
       target (np.array): The true labels.
       predicted (np.array): The predicted probabilities.
       threshold (float): The threshold to use for classification.

    Returns:
       sensitivity (float): The sensitivity for the given threshold.
       specificity (float): The specificity for the given threshold.
    """
    # Apply the threshold to the predicted probabilities
    thresh_preds = np.zeros(len(predicted))
    thresh_preds[predicted > threshold] = 1

    # Calculate the confusion matrix
    cm = metrics.confusion_matrix(target, thresh_preds)

    # Calculate the sensitivity and specificity
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    return sensitivity, specificity


def nroc_curve(y_true: np.array, predicted: np.array, num_thresh: int = 1000) -> tuple:
    """
    Calculate the false positive rates and true positive rates for different thresholds to generate a ROC curve.

    Parameters:
       y_true (np.array): The true labels.
       predicted (np.array): The predicted probabilities.
       num_thresh (int): The number of thresholds to use.

    Returns:
       fprs (list): The false positive rates for the thresholds.
       tprs (list): The true positive rates for the thresholds.
       thresholds (np.array): The thresholds used.
    """
    step = 1 / num_thresh
    thresholds = np.arange(0, 1 + step, step)
    fprs, tprs = [], []

    for threshold in thresholds:
        y_pred = np.where(predicted >= threshold, 1, 0)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fprs.append(fp / (fp + tn))
        tprs.append(tp / (tp + fn))

    return fprs, tprs, thresholds

