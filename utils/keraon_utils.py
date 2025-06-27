#!/usr/bin/python
# Robert Patton, rpatton@fredhutch.org
# v0.1, 5/7/2025

import pandas as pd
import numpy as np
import os
import re
import random
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Tuple


def load_reference_key(key_path):
    try:
        with open(key_path, 'r') as f:
            first_line = f.readline().strip()
        
        if not first_line:
            print(f"Error: Reference key file '{key_path}' is empty. Exiting.")
            exit()

        first_line_parts = first_line.split('\t')
        
        if len(first_line_parts) != 3:
            print(f"Error: Reference key file '{key_path}' must have exactly 3 tab-seperated columns (sample, subtype, purity). Found {len(first_line_parts)} in the first line: '{first_line}'. Exiting.")
            exit()

        has_header = False
        try:
            float(first_line_parts[2]) # Check if third element of first line is numeric
        except ValueError:
            has_header = True # If not numeric, it's a header

        if has_header:
            print(f"Header detected in reference key file. Header: {first_line_parts}")
            ref_labels = pd.read_table(key_path, sep='\t', index_col=0, header=0)
        else:
            print("No header detected in reference key file. Assigning default column names.")
            ref_labels = pd.read_table(key_path, sep='\t', index_col=0, header=None)

        # Validate number of columns after loading (index_col=0 means 2 remaining columns)
        if ref_labels.shape[1] != 2:
            print(f"Error: Reference key file '{key_path}' should have exactly 3 tab-seperated columns (sample, subtype, purity). After loading with first column as index, expected 2 data columns, but got {ref_labels.shape[1]}. Exiting.")
            exit()
        
        # Assign standard column names
        original_columns = list(ref_labels.columns)
        ref_labels.columns = ['Subtype', 'Purity']
        # print(f"Columns assigned as: Index (from first file column), 'Subtype' (from second file column, originally '{original_columns[0]}'), 'Purity' (from third file column, originally '{original_columns[1]}')")

        # Check if 'Purity' column is numeric
        if not pd.api.types.is_numeric_dtype(ref_labels['Purity']):
            print(f"Attempting to convert 'Purity' column (original name: '{original_columns[1]}') to numeric.")
            ref_labels['Purity'] = pd.to_numeric(ref_labels['Purity'], errors='coerce')
            if ref_labels['Purity'].isnull().any():
                print(f"Error: The 'Purity' column (third column in '{key_path}', originally '{original_columns[1]}') contains non-numeric values that could not be converted. Exiting.")
                exit()
            print("'Purity' column successfully converted to numeric.")

    except FileNotFoundError:
        print(f"Error: Reference key file '{key_path}' not found. Exiting.")
        exit()
    except pd.errors.EmptyDataError:
        print(f"Error: Reference key file '{key_path}' is empty or malformed after attempting to read. Exiting.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading reference key file '{key_path}': {e}. Exiting.")
        exit()

    # List number of unique examples of each Subtype and filter
    subtype_counts = ref_labels['Subtype'].value_counts()
    print("\nNumber of unique examples per Subtype before filtering:")
    if subtype_counts.empty:
        print("No subtypes found in the reference key file.")
    else:
        for subtype, count in subtype_counts.items():
            print(f"- {subtype}: {count}")

    subtypes_to_drop = subtype_counts[subtype_counts < 3].index.tolist()
    
    if subtypes_to_drop:
        print("\nFiltering subtypes with fewer than 3 examples:")
        samples_dropped_details = []
        for subtype in subtypes_to_drop:
            count = subtype_counts[subtype]
            print(f"Subtype '{subtype}' has {count} examples, which is less than the 3 minimum. Samples for this subtype will be dropped.")
            samples_for_subtype = ref_labels[ref_labels['Subtype'] == subtype].index.tolist()
            for sample_id in samples_for_subtype:
                samples_dropped_details.append(f"Sample '{sample_id}' (Subtype: {subtype})")

        ref_labels = ref_labels[~ref_labels['Subtype'].isin(subtypes_to_drop)]
        
        if samples_dropped_details:
            print("\nDropped samples:")
            for detail in samples_dropped_details:
                print(f"- {detail}")
        
        if ref_labels.empty:
            print("\nWarning: All samples were dropped after filtering. The reference dataset is now empty. This may cause issues downstream.")
        else:
            print("\nNumber of unique examples per Subtype after filtering:")
            new_subtype_counts = ref_labels['Subtype'].value_counts()
            if new_subtype_counts.empty:
                print("No subtypes remaining after filtering.")
            else:
                for subtype, count in new_subtype_counts.items():
                    print(f"- {subtype}: {count}")

    # Check if 'Healthy' subtype is present
    if "Healthy" not in ref_labels['Subtype'].values:
        print("Warning: 'Healthy' subtype not found in the reference key. Please double-check your reference key file. Exiting.")
        exit()
    
    if ref_labels.empty and not subtypes_to_drop: # Handles case where file was empty or became empty before filtering step
        print("Error: Reference labels are empty. Cannot proceed. Exiting.")
        exit()
    
    return ref_labels


def load_test_labels(file_path: str):
    """
    Loads test labels (TFX and optional truth values) from a file.

    The file is expected to be tab-separated and can optionally have a header row.
    It must contain 2 or 3 columns of data after any header.
    - Column 1: Sample identifiers (will become the DataFrame index).
    - Column 2: Tumor Fraction (TFX) values, must be numeric.
    - Column 3 (optional): True subtype/category labels (strings).

    Args:
        file_path (str): The path to the input file.

    Returns:
        tuple: A tuple containing two elements:
            - test_labels_df (pd.DataFrame): DataFrame with sample identifiers as the index
              and a single column 'TFX' containing numeric tumor fraction values.
            - truth_vals_df (pd.DataFrame or None): DataFrame with sample identifiers as
              the index and a single column 'Truth' containing string labels.
              Returns None if the input file does not have a third column.
    
    Raises:
        SystemExit: If the file is not found, improperly formatted, or contains
                    invalid data that prevents processing according to the rules.
    """
    try:
        # Read with no header initially, assuming tab separation.
        # dtype=str ensures all data is read as string initially.
        # keep_default_na=False and na_filter=False prevent "NA" or empty fields from becoming NaN.
        df_initial = pd.read_csv(
            file_path, 
            sep='\t', 
            header=None, 
            comment='#', 
            skip_blank_lines=True, 
            dtype=str, 
            keep_default_na=False, 
            na_filter=False
        )
        # Remove rows that are entirely empty
        df_initial = df_initial.dropna(how='all').reset_index(drop=True)

        if df_initial.empty:
            print(f"Error: File at {file_path} is empty or contains only comments/blank lines.")
            exit(1)

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: File at {file_path} is empty (pd.errors.EmptyDataError).")
        exit(1)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        exit(1)

    df_data = df_initial
    first_row_is_header = False

    # Determine if the first row is a header by checking if the second column's first entry is non-numeric
    if df_initial.shape[1] >= 2: # Need at least two columns to check the second one
        try:
            float(df_initial.iloc[0, 1]) # Try to convert second item of first row to float
        except ValueError:
            first_row_is_header = True # Failed conversion suggests it's a header
        except IndexError:
            # This should not be reached if df_initial is not empty and shape[1] >= 2
            print(f"Error: Unexpected IndexError while checking for header in {file_path}.")
            exit(1)
    
    if first_row_is_header:
        df_data = df_initial.iloc[1:].reset_index(drop=True)
        if df_data.empty:
            print(f"Error: File {file_path} contained only a header row or became empty after removing header.")
            exit(1)
    
    num_cols = df_data.shape[1]

    if num_cols not in [2, 3]:
        print(f"Error: File {file_path} (after potential header removal) must have 2 or 3 data columns, but found {num_cols}.")
        exit(1)

    # Column 1: Samples (Index)
    if df_data.iloc[:, 0].astype(str).eq('').any():
        print(f"Error: The first column (sample identifiers) in {file_path} contains empty strings.")
        exit(1)
    samples = df_data.iloc[:, 0].astype(str)

    # Column 2: TFX (Numeric)
    try:
        tfx_values_series = pd.to_numeric(df_data.iloc[:, 1], errors='coerce')
        if tfx_values_series.isnull().any():
            problematic_indices = tfx_values_series[tfx_values_series.isnull()].index
            problematic_original_values = df_data.iloc[problematic_indices, 1]
            print(f"Error: The second column (TFX) in {file_path} contains non-numeric values.")
            print(f"Problematic original values at data rows (0-indexed after header removal): {problematic_indices.tolist()}:\n{problematic_original_values.to_string()}")
            exit(1)
    except Exception as e:
        print(f"Error: Could not convert the second column (TFX) in {file_path} to numeric. Original error: {e}")
        exit(1)

    test_labels_df = pd.DataFrame({'TFX': tfx_values_series.values}, index=samples)
    test_labels_df.index.name = 'Sample'

    truth_vals_df = None
    if num_cols == 3:
        truth_values_series = df_data.iloc[:, 2].astype(str)
        truth_vals_df = pd.DataFrame({'Truth': truth_values_series.values}, index=samples)
        truth_vals_df.index.name = 'Sample'

    return test_labels_df, truth_vals_df


def is_hex_color(color_string: str) -> bool:
    """Checks if a string is a valid hex color code (e.g., #RRGGBB or #RGB)."""
    if not isinstance(color_string, str):
        return False
    pattern = re.compile(r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$')
    return bool(pattern.match(color_string))


def load_palette(palette_path: str, ref_labels: pd.DataFrame = None) -> dict:
    """
    Loads a palette from a file or generates a default one.
    Ensures 'Healthy' is '#009988' and all ref_labels subtypes are covered.
    """
    healthy_color = "#009988"
    palette = {}
    if ref_labels is None:
        print("Warning: No reference labels provided. Palette will not be validated against subtypes.")
        all_ref_subtypes = set()
    else:
        print(f"Reference labels provided. Palette will be validated against {len(ref_labels)} subtypes.")
        all_ref_subtypes = set(ref_labels['Subtype'].unique())

    if palette_path is None:
        print("\nPalette file not provided. Generating default palette.")
        # Color-blind friendly colors
        default_colors = [
            "#E69F00", "#56B4E9", "#F0E442", "#0072B2", 
            "#D55E00", "#CC79A7", "#AA4499", "#DDCC77", 
            "#88CCEE", "#44AA99", "#BBBBBB"
        ]
        
        # Ensure "Healthy" gets its specific color
        palette["Healthy"] = healthy_color
        
        color_idx = 0
        for subtype in sorted(list(all_ref_subtypes)): # Sort for consistent color assignment
            if subtype not in palette: # If not already "Healthy"
                if color_idx < len(default_colors):
                    palette[subtype] = default_colors[color_idx]
                    color_idx += 1
                else:
                    # Fallback to random color if out of predefined ones
                    random_color = "#%06X" % random.randint(0, 0xFFFFFF)
                    palette[subtype] = random_color
                    print(f"Warning: Ran out of predefined default colors. Assigning random color {random_color} to '{subtype}'.")
        print("Default palette generated.")

    else:
        print(f"\nLoading palette from: {palette_path}")
        try:
            with open(palette_path, 'r') as f:
                first_line = f.readline().strip()
            
            if not first_line:
                print(f"Error: Palette file '{palette_path}' is empty. Exiting.")
                exit(1)

            first_line_parts = first_line.split('\t')
            if len(first_line_parts) != 2:
                print(f"Error: Palette file '{palette_path}' must have exactly 2 columns. Found {len(first_line_parts)} in the first line: '{first_line}'. Exiting.")
                exit(1)

            has_header = not is_hex_color(first_line_parts[1])

            if has_header:
                print(f"Header detected in palette file. Header: {first_line_parts}")
                palette_df = pd.read_csv(palette_path, sep='\t', header=0, dtype=str)
            else:
                print("No header detected in palette file.")
                palette_df = pd.read_csv(palette_path, sep='\t', header=None, dtype=str)

            if palette_df.shape[1] != 2:
                print(f"Error: Palette file '{palette_path}' should have 2 columns. Found {palette_df.shape[1]}. Exiting.")
                exit(1)
            
            palette_df.columns = ['Subtype', 'Color']
            
            for index, row in palette_df.iterrows():
                subtype_name = str(row['Subtype']).strip()
                color_val = str(row['Color']).strip()
                if not is_hex_color(color_val):
                    print(f"Error: Invalid hex color format '{color_val}' for subtype '{subtype_name}' in palette file '{palette_path}' at row {index + (1 if has_header else 0) + 1}. Exiting.")
                    exit(1)
                palette[subtype_name] = color_val.upper()
            
            print("Palette file loaded.")

            # Check if all ref_labels subtypes are in the palette
            missing_subtypes = all_ref_subtypes - set(palette.keys())
            if missing_subtypes:
                print(f"Error: The following subtypes from the reference key are missing in the palette file '{palette_path}': {', '.join(missing_subtypes)}. Please add them. Exiting.")
                exit(1)

        except FileNotFoundError:
            print(f"Error: Palette file '{palette_path}' not found. Exiting.")
            exit(1)
        except pd.errors.EmptyDataError:
            print(f"Error: Palette file '{palette_path}' is empty or malformed after attempting to read. Exiting.")
            exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading palette file '{palette_path}': {e}. Exiting.")
            exit(1)

    # Ensure "Healthy" color is correct and present
    if "Healthy" in palette:
        if palette["Healthy"].upper() != healthy_color.upper():
            print(f"Warning: 'Healthy' subtype color in palette ('{palette['Healthy']}') is not the standard '{healthy_color}'. Overriding to '{healthy_color}'.")
            palette["Healthy"] = healthy_color
    else:
        # Add Healthy if it wasn't in the file or generated by default (e.g. not in ref_labels for default)
        # This ensures "Healthy" is always in the palette object if this function is called.
        palette["Healthy"] = healthy_color
        if palette_path is not None: # Only print if it was expected from a file
             print(f"Info: 'Healthy' subtype was not found in the palette file. Added with default color '{healthy_color}'.")
        elif "Healthy" not in all_ref_subtypes: # If default generation and Healthy wasn't a ref_subtype
             print(f"Info: Adding 'Healthy' with default color '{healthy_color}' to the palette.")


    print("Palette processing complete.")
    return palette


def _plot_feature_distribution(data_df: pd.DataFrame, 
                               feature_type_to_plot: str, 
                               current_palette: dict, 
                               plot_destination_subdir: str, 
                               plot_filename_suffix: str, 
                               plot_title_prefix: str,
                               input_file_description_for_title: str,
                               ref_labels_for_coloring: pd.DataFrame = None):
    """Helper function to plot feature distributions."""
    plt.figure(figsize=(10, 6))
    
    # Data for the current feature type
    feature_data_subset = data_df[data_df['feature'] == feature_type_to_plot]

    if feature_data_subset.empty:
        print(f"  - No data for feature '{feature_type_to_plot}' at this stage, skipping plot: {plot_filename_suffix}.")
        plt.close()
        return

    title = f"{plot_title_prefix} of {feature_type_to_plot} {input_file_description_for_title}"

    if ref_labels_for_coloring is not None:
        temp_ref_labels = ref_labels_for_coloring.copy()
        if temp_ref_labels.index.name != 'sample':
            temp_ref_labels.index.name = 'sample'
        
        plot_data_merged = pd.merge(feature_data_subset, temp_ref_labels[['Subtype']],
                                    left_on='sample', right_index=True, how='inner')

        if plot_data_merged.empty or 'Subtype' not in plot_data_merged.columns or plot_data_merged['Subtype'].isnull().all():
            print(f"  - Warning: No subtype data for feature '{feature_type_to_plot}' after merging for plot '{plot_filename_suffix}'. Plotting global distribution.")
            sns.histplot(data=feature_data_subset, x='value', kde=True, stat="density", common_norm=False)
        else:
            subtypes_in_plot_data = plot_data_merged['Subtype'].unique()
            for subtype in subtypes_in_plot_data:
                subtype_specific_data = plot_data_merged[plot_data_merged['Subtype'] == subtype]
                color = current_palette.get(subtype, "#333333")
                sns.histplot(subtype_specific_data['value'], kde=True, label=f"{subtype} (n={len(subtype_specific_data)})",
                             color=color, stat="density", common_norm=False, element="step", alpha=0.7)
            plt.legend(title="Subtype", bbox_to_anchor=(1.05, 1), loc='upper left')
    else: # No ref_labels for coloring, plot global histogram and KDE
        sns.histplot(feature_data_subset['value'], kde=False, stat="density", color="grey", alpha=0.5, label="Overall Distribution")
        # Plot a single global KDE line
        sns.kdeplot(feature_data_subset['value'], color="blue", linewidth=1.5, label="Global KDE")
        
        plt.legend(title="Distribution", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.title(title)
    plt.xlabel(f"{feature_type_to_plot} Value")
    plt.ylabel("Density")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    clean_feature_type = feature_type_to_plot.replace(' ', '_').replace('/', '_')
    plot_filename = os.path.join(plot_destination_subdir, f"{clean_feature_type}{plot_filename_suffix}.pdf")
    try:
        plt.savefig(plot_filename)
    except Exception as e_plot:
        print(f"  - Error saving plot {plot_filename}: {e_plot}")
    plt.close()
    # print(f"  - Saved plot: {plot_filename}")


def load_triton_fm(fm_path: Union[str, List[str]], 
                   scaling_methods: Dict[str, callable], 
                   output_dir_for_plots: str, 
                   palette: dict, 
                   ref_labels: pd.DataFrame = None, 
                   plot_distributions: bool = True,
                   limit_features: list = None,
                   feature_scaling_params: Dict = None) -> Tuple[pd.DataFrame, Union[Dict, None]]:
    """
    Loads Triton feature matrix(es), applies initial scaling, optionally calculates/applies
    feature-wise scaling, plots distributions, and returns a pivoted DataFrame
    and scaling parameters if calculated.
    """
    # 1. Load and Validate
    if isinstance(fm_path, str):
        fm_paths = [fm_path]
        input_fm_description_for_title = f"from {os.path.basename(fm_path)}"
    elif isinstance(fm_path, list):
        fm_paths = fm_path
        input_fm_description_for_title = "from combined files"
        if not fm_paths:
            print("Error: fm_path list is empty. Exiting.")
            exit(1)
    else:
        print(f"Error: fm_path must be a string or a list of strings. Got {type(fm_path)}. Exiting.")
        exit(1)

    # print(f"\nLoading Triton feature matrix/matrices from: {fm_paths}")
    
    all_dfs = []
    for current_path in fm_paths:
        print(f"Processing: {current_path}")
        try:
            with open(current_path, 'r') as f: header_line = f.readline().strip()
            header_parts = header_line.split('\t')
            expected_header = ["sample", "site", "feature", "value"]
            df_peek = pd.read_csv(current_path, sep='\t', nrows=1)
            if df_peek.shape[1] != 4:
                print(f"Error: FM '{current_path}' must have 4 columns. Found {df_peek.shape[1]}. Exiting.")
                exit(1)
            if header_parts != expected_header:
                print(f"Error: FM '{current_path}' incorrect header. Expected: {expected_header}. Found: {header_parts}. Exiting.")
                exit(1)
            current_df = pd.read_csv(current_path, sep='\t', header=0, dtype={'sample': str, 'site': str, 'feature': str})
            current_df['value'] = pd.to_numeric(current_df['value'], errors='coerce') # Coerce will turn unconvertibles to NaN
            
            all_dfs.append(current_df)
            print(f"FM '{current_path}' loaded ({len(current_df)} rows).")
        except FileNotFoundError: print(f"Error: FM '{current_path}' not found. Exiting."); exit(1)
        except pd.errors.EmptyDataError: print(f"Error: FM '{current_path}' is empty. Exiting."); exit(1)
        except Exception as e: print(f"Error loading FM '{current_path}': {e}. Exiting."); exit(1)

    if not all_dfs: print("Error: No data loaded. Exiting."); exit(1)
    df = pd.concat(all_dfs, ignore_index=True)

    # Restrict to specific features if limit_features is provided
    if limit_features:
        print(f"\nApplying feature limiting based on `limit_features` list: {limit_features}")
        df = df[df['feature'].isin(limit_features)]
    
    # Drop rows with NaN in essential columns, especially 'value' after to_numeric coerce
    initial_row_count = len(df)
    df.dropna(subset=['sample', 'site', 'feature', 'value'], inplace=True)
    if len(df) < initial_row_count:
        print(f"Dropped {initial_row_count - len(df)} rows with NaN values in essential columns (sample, site, feature, value).")
    if df.empty: print("Error: DataFrame empty after NaN drop. Exiting."); exit(1)


    # 2. Apply Initial Scaling (scaling_methods)
    print("\nApplying initial/default feature-level scaling_methods...")
    scaled_features_applied_count = 0
    for feature_name_in_dict, scaling_func in scaling_methods.items():
        mask = df['feature'] == feature_name_in_dict
        if mask.any():
            df.loc[mask, 'value'] = df.loc[mask, 'value'].apply(scaling_func)
            print(f" - Initial scaling for '{feature_name_in_dict}' applied to {mask.sum()} rows.")
            scaled_features_applied_count += 1
    if scaled_features_applied_count == 0: print("No initial scaling_methods applied (no matching features or empty dict).")
    else: print(f"Initial scaling_methods applied to {scaled_features_applied_count} distinct feature types.")

    # Determine plot subdirectory base name
    plot_base_dir_name = "reference_features" if ref_labels is not None else "test_features"
    overall_plot_subdir = os.path.join(output_dir_for_plots, "feature_distributions", plot_base_dir_name)
    os.makedirs(overall_plot_subdir, exist_ok=True)

    # 3. Feature-wise Scaling
    print("\nApplying robust centering (site-specific Healthy medians) and feature-level robust scaling ...")

    scaling_params = {}

    if ref_labels is not None:  # Reference Run: Calculate and apply scaling
        # Identify healthy samples (assumes 'Healthy' is in ref_labels['Subtype'], case-insensitive)
        healthy_samples = ref_labels[ref_labels['Subtype'].str.lower() == 'healthy'].index
        # Compute, for each (site, feature) combination, the median value among healthy samples
        healthy_centers = df[df['sample'].isin(healthy_samples)].groupby(['site', 'feature'])['value'].median()
        scaling_params['centers'] = healthy_centers.to_dict()
        
        # Subtract the healthy center from every row for its (site, feature)
        def apply_center(row):
            key_tuple = (row['site'], row['feature'])
            center = healthy_centers.get(key_tuple, 0)  # if missing, assume 0
            return row['value'] - center
        df['centered_value'] = df.apply(apply_center, axis=1)
        
        # Compute feature-level robust scaling factors using the IQR of the centered values (across all sites)
        robust_scales = df.groupby('feature')['centered_value'].agg(lambda x: x.quantile(0.75) - x.quantile(0.25))
        # Avoid division by zero: if IQR is 0, default to 1
        robust_scales = robust_scales.replace(0, 1)
        scaling_params['scale_factors'] = robust_scales.to_dict()
        
        # Apply scaling for each row
        def apply_scaling(row):
            scale = scaling_params['scale_factors'].get(row['feature'], 1)
            return row['centered_value'] / scale
        df['scaled_value'] = df.apply(apply_scaling, axis=1)
        df['value'] = df['scaled_value']
        
        print("Reference run: Robust centering and feature-level scaling applied.")
        params_to_return = scaling_params

    else:  # Test Run: Apply provided scaling parameters
        if feature_scaling_params is None:
            print("Error: Running in test mode but no scaling parameters provided. Exiting.")
            exit(1)
        
        scaling_params = feature_scaling_params  # Expected to contain keys 'centers' and 'scale_factors'
        
        def apply_center_test(row):
            key_tuple = (row['site'], row['feature'])
            center = scaling_params.get('centers', {}).get(key_tuple, 0)
            return row['value'] - center
        df['centered_value'] = df.apply(apply_center_test, axis=1)
        
        def apply_scaling_test(row):
            scale = scaling_params.get('scale_factors', {}).get(row['feature'], 1)
            return row['centered_value'] / scale
        df['scaled_value'] = df.apply(apply_scaling_test, axis=1)
        df['value'] = df['scaled_value']
        
        print("Test run: Applied provided robust centering and feature-level scaling.")
        params_to_return = None

    # 4. Filter Samples based on ref_labels (if provided, AFTER all scaling)
    if ref_labels is not None:
        print("\nFiltering feature matrix samples based on reference labels (post-scaling)...")
        ref_sample_ids = set(ref_labels.index)
        fm_sample_ids_before_filter = set(df['sample'].unique())
        missing_ref_samples_in_fm = ref_sample_ids - fm_sample_ids_before_filter
        if missing_ref_samples_in_fm:
            print(f"Error: Samples from ref_labels NOT found in FM data: {list(missing_ref_samples_in_fm)}. Exiting.")
            exit(1)
        else:
            print("All samples from ref_labels are present in the FM data.")
        
        df = df[df['sample'].isin(ref_sample_ids)]
        fm_sample_ids_after_filter = set(df['sample'].unique())
        dropped_samples_count = len(fm_sample_ids_before_filter) - len(fm_sample_ids_after_filter)
        if dropped_samples_count > 0:
            print(f"Dropped {dropped_samples_count} samples from FM not in ref_labels.")
        else:
            print("No samples dropped from FM based on ref_labels (all FM samples relevant).")
        if df.empty: print("Error: FM empty after filtering with ref_labels. Exiting."); exit(1)
        print(f"FM now contains {len(fm_sample_ids_after_filter)} samples matching ref_labels.")

    # Plot after feature-wise scaling
    if plot_distributions:
        print(f"\nGenerating plots after feature-wise scaling in: {overall_plot_subdir}")
        # Determine which ref_labels to use for coloring Plot 2
        # If it was a reference run, ref_labels was used for filtering, so it's the correct one.
        # If it was a test run, ref_labels is None, so no coloring by subtype.
        ref_labels_for_plot2 = ref_labels if ref_labels is not None else None
        
        for feature_type in df['feature'].unique():
            _plot_feature_distribution(df, feature_type, palette, overall_plot_subdir,
                                       "_post_scaling", "Distribution After Scaling",
                                       input_fm_description_for_title,
                                       ref_labels_for_coloring=ref_labels_for_plot2)

    # Final processing: clean site/feature names and pivot
    print("\nPivoting feature matrix...")
    df['site'] = df['site'].str.replace('_', '-')
    df['feature'] = df['feature'].str.replace('_', '-')
    df['cols'] = df['site'].astype(str) + '_' + df['feature']
    
    if df.duplicated(subset=['sample', 'cols']).any():
        num_duplicates = df.duplicated(subset=['sample', 'cols']).sum()
        print(f"Warning: Found {num_duplicates} duplicate 'sample'-'cols' combinations. Aggregating by mean.")
        df_pivoted = df.pivot_table(index='sample', columns='cols', values='value', aggfunc='mean')
    else:
        df_pivoted = df.pivot_table(index='sample', columns='cols', values='value')
    
    print(f"Pivoting complete. Resulting DataFrame shape: {df_pivoted.shape}")
    return df_pivoted, params_to_return

