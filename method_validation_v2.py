import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="Analytical Method Validation Analyzer",
    page_icon="🔬",
    layout="wide"
)

# --- Functions ---

def calculate_summary_stats(df, compound, group_by_cols):
    """
    Calculates and returns summary statistics for a given compound.
    """
    # Ensure the compound column is numeric, coercing errors to NaN
    df[compound] = pd.to_numeric(df[compound], errors='coerce')
    # Drop rows where the compound measurement is not a number
    df = df.dropna(subset=[compound])

    if df.empty:
        return pd.DataFrame() # Return empty dataframe if no valid data

    # Use named aggregation for clearer column names from the start
    summary = df.groupby(group_by_cols).agg(
        Mean=(compound, 'mean'),
        SD=(compound, 'std'),
        Min=(compound, 'min'),
        Max=(compound, 'max'),
        N=(compound, 'count')
    ).reset_index()

    # Calculate derived statistics, handling potential division by zero
    summary['%RSD'] = 100 * (summary['SD'] / summary['Mean']).replace([float('inf'), -float('inf')], None)
    summary['Mean % Recovery'] = 100 * (summary['Mean'] / summary['Level']).replace([float('inf'), -float('inf')], None)
    # --- New Calculation for Expanded Uncertainty ---
    summary['Uexp (k=2)'] = 2 * summary['SD']

    return summary

def create_level_specific_boxplot(df_level, compounds, level_value):
    """
    Creates a boxplot for a specific level, comparing all selected compounds.
    The X-axis shows the compounds.
    """
    # --- 1. Reshape data from wide to long format ---
    id_vars = ['Level', 'Analyst', 'Date', 'Sample']
    id_vars_present = [col for col in id_vars if col in df_level.columns]
    
    long_df = df_level.melt(
        id_vars=id_vars_present,
        value_vars=compounds,
        var_name='Compound',
        value_name='Measurement'
    )

    # --- 2. Data Cleaning and Calculation ---
    long_df['Measurement'] = pd.to_numeric(long_df['Measurement'], errors='coerce')
    long_df['Level'] = pd.to_numeric(long_df['Level'], errors='coerce')
    long_df = long_df.dropna(subset=['Measurement', 'Level'])

    # Calculate % Recovery for each individual data point
    long_df['% Recovery'] = 100 * (long_df['Measurement'] / long_df['Level']).replace([float('inf'), -float('inf')], None)
    long_df = long_df.dropna(subset=['% Recovery'])
    
    if long_df.empty:
        return None # Return None if no data to plot

    # --- 3. Create the box plot for the specific level ---
    fig = px.box(
        long_df,
        x='Compound',  # X-axis is now Compound
        y='% Recovery',
        color='Compound', # Assigns a unique color to each compound's boxplot
        title=f'% Recovery Distribution at Level {level_value}',
        labels={
            'Compound': 'Compound', # Label for the X-axis
            '% Recovery': '% Recovery'
        },
        points="all",
        hover_data=['Analyst', 'Date', 'Sample']
    )
    # Add space between the boxplots for better readability
    fig.update_layout(boxgroupgap=0.5, boxgap=0.3)
    return fig

def generate_validation_report(summary_df, criteria):
    """
    Compares summary statistics against validation criteria and generates a PASS/FAIL report.
    """
    # Create the base of the report from the summary
    report_df = summary_df[['Compound', 'Level']].copy()
    
    # --- Get criteria values from the dictionary, defaulting to None if not found ---
    rsd_max = criteria.get('%RSD max', None)
    rec_max = criteria.get('Mean % Recovery max', None)
    rec_min = criteria.get('Mean % Recovery min', None)

    # --- Perform checks and generate PASS/FAIL status for each criterion ---
    
    # 1. %RSD Check
    if rsd_max is not None:
        # Compare each row's %RSD against the criterion
        report_df['RSD Check'] = summary_df.apply(
            lambda row: "PASS" if pd.notna(row['%RSD']) and row['%RSD'] <= rsd_max else "FAIL",
            axis=1
        )
    else:
        report_df['RSD Check'] = "N/A"

    # 2. Mean % Recovery Max Check
    if rec_max is not None:
        report_df['Recovery Max Check'] = summary_df.apply(
            lambda row: "PASS" if pd.notna(row['Mean % Recovery']) and row['Mean % Recovery'] <= rec_max else "FAIL",
            axis=1
        )
    else:
        report_df['Recovery Max Check'] = "N/A"

    # 3. Mean % Recovery Min Check
    if rec_min is not None:
        report_df['Recovery Min Check'] = summary_df.apply(
            lambda row: "PASS" if pd.notna(row['Mean % Recovery']) and row['Mean % Recovery'] >= rec_min else "FAIL",
            axis=1
        )
    else:
        report_df['Recovery Min Check'] = "N/A"

    # --- Determine the overall status ---
    check_cols = [col for col in report_df.columns if 'Check' in col]
    
    def get_overall_status(row):
        # If any check is marked as "FAIL", the overall status is "FAIL"
        if "FAIL" in row[check_cols].values:
            return "FAIL"
        # If no checks failed, the overall status is "PASS"
        return "PASS"

    report_df['Overall Status'] = report_df.apply(get_overall_status, axis=1)

    return report_df

def style_report(val):
    """
    Applies color to PASS/FAIL/N/A cells for better visualization.
    """
    if val == "PASS":
        color = 'green'
    elif val == "FAIL":
        color = 'red'
    else: # For "N/A"
        color = 'grey'
    return f'color: {color}; font-weight: bold;'


# --- Main Application ---
st.title("🔬 Analytical Method Validation Data Analyzer")
st.markdown("""
Welcome! This application helps you analyze data for analytical method validation.

**Instructions:**
1.  Upload your data as an Excel file (`.xlsx`).
2.  The Excel file must contain a sheet named `Validation data`.
3.  Ensure your data columns match the required format (see column descriptions below).
4.  Use the sidebar to select compounds and grouping options for analysis.

**Required Columns in 'Validation data' sheet:**
- `Index`, `Date`, `Analyst`, `Sample`, `Units`, `Level`, `Notes`
- From column H onwards: Measurement data for each compound (numeric).
""")

# --- File Uploader ---
uploaded_file = st.file_uploader(
    "Choose your Excel file (Max 100MB)",
    type="xlsx",
    help="Upload the Excel file containing the 'Validation data' sheet."
)

if uploaded_file is not None:
    try:
        # Read the main data and criteria sheets from the uploaded excel file
        df = pd.read_excel(uploaded_file, sheet_name="Validation data")
        
        criteria_dict = None
        try:
            criteria_df = pd.read_excel(uploaded_file, sheet_name="Criteria", header=None)
            # Convert the two-column criteria sheet into a dictionary
            # It takes column 0 as keys and column 1 as values
            criteria_df[0] = criteria_df[0].str.strip() # Clean up whitespace
            criteria_dict = criteria_df.set_index(0)[1].to_dict()
            st.sidebar.success("Successfully loaded 'Criteria' sheet.")
        except Exception:
            st.sidebar.warning("Optional 'Criteria' sheet not found. PASS/FAIL report will not be generated.")


        # --- Data Validation ---
        required_cols = ["Index", "Date", "Analyst", "Sample", "Units", "Level", "Notes"]
        
        # --- Normalization and Validation Logic ---
        # Create a mapping from the standardized lowercase name to the original name in the file
        # This handles both whitespace and case-insensitivity
        cols_in_file = {c.strip().lower(): c for c in df.columns}
        
        missing_cols = []
        # This will hold the mapping from the original file's column name to our standard name
        # e.g., {'  iNdEx  ': 'Index', 'LEVEL': 'Level'}
        rename_map = {}

        for req_col in required_cols:
            if req_col.lower() in cols_in_file:
                # Get the original column name from the file (e.g., "  iNdEx  ")
                original_col_name = cols_in_file[req_col.lower()]
                # Map it to our desired standard name (e.g., "Index")
                rename_map[original_col_name] = req_col
            else:
                missing_cols.append(req_col)

        if missing_cols:
            st.error(f"Error: The uploaded file is missing one or more required columns. Please ensure the following columns exist: **{', '.join(missing_cols)}**")
            st.info(f"We detected the following columns in your file: {', '.join(df.columns)}")
        else:
            # If all columns are found, rename them to the standard casing for consistency
            df.rename(columns=rename_map, inplace=True)

            # Identify compound columns (all columns after "Notes")
            notes_index = df.columns.get_loc("Notes")
            compound_columns = df.columns[notes_index + 1:].tolist()

            if not compound_columns:
                st.warning("No compound data columns found after the 'Notes' column.")
            else:
                st.success(f"File uploaded successfully! Found {len(compound_columns)} compound(s): {', '.join(compound_columns)}")

                # --- Sidebar for User Input ---
                st.sidebar.header("Analysis Options")
                selected_compounds = st.sidebar.multiselect(
                    "Select compounds to analyze:",
                    options=compound_columns,
                    default=compound_columns
                )

                grouping_option = st.sidebar.radio(
                    "Select additional grouping for statistics (optional):",
                    options=["None", "Analyst", "Sample"],
                    index=0
                )

                # --- Analysis and Plotting ---
                if not selected_compounds:
                    st.info("Please select at least one compound from the sidebar to start the analysis.")
                else:
                    st.markdown("---")
                    st.header("Combined Analysis")

                    # --- 1. Consolidated Summary Statistics ---
                    with st.expander("View Summary Statistics for All Selected Compounds", expanded=True):
                        all_summaries = []
                        group_by_cols = ['Level']
                        if grouping_option != "None":
                            group_by_cols.append(grouping_option)

                        for compound in selected_compounds:
                            summary_df = calculate_summary_stats(df, compound, group_by_cols)
                            if not summary_df.empty:
                                summary_df['Compound'] = compound
                                all_summaries.append(summary_df)
                        
                        if not all_summaries:
                            st.warning("No valid numerical data to calculate statistics for the selected compounds.")
                        else:
                            # Combine all individual summaries into one DataFrame
                            final_summary_df = pd.concat(all_summaries, ignore_index=True)
                            
                            # Reorder columns to have Compound first
                            cols_order = ['Compound'] + [col for col in final_summary_df if col != 'Compound']
                            final_summary_df = final_summary_df[cols_order]

                            st.write(f"Statistics grouped by **{' and '.join(group_by_cols)}**:")
                            st.dataframe(final_summary_df.style.format({
                                'Mean': '{:.3f}',
                                'SD': '{:.3f}',
                                'Min': '{:.3f}',
                                'Max': '{:.3f}',
                                'N': '{}',
                                '%RSD': '{:.2f}%',
                                'Mean % Recovery': '{:.2f}%',
                                'Uexp (k=2)': '{:.3f}'
                            }), use_container_width=True)

                    # --- 2. Expanded Uncertainty Report ---
                    st.header("Expanded Uncertainty Report")
                    with st.expander("View Uncertainty Details", expanded=True):
                        st.markdown("""
                        The Expanded Uncertainty (`Uexp`) provides an interval where the true value is believed to lie with a high level of confidence (approximately 95%).
                        - It is estimated using a coverage factor of **k=2**.
                        - **Calculation**: `Uexp = 2 * Standard Deviation (SD)`
                        """)
                        
                        # Create and display the dedicated uncertainty report
                        uncertainty_report_df = final_summary_df[['Compound', 'Level', 'Uexp (k=2)']]
                        
                        st.dataframe(
                            uncertainty_report_df.style.format({'Uexp (k=2)': '{:.3f}'}),
                            use_container_width=True
                        )

                    # --- 3. Validation Report (if criteria are available) ---
                    if criteria_dict:
                        st.header("Validation Report")
                        validation_report = generate_validation_report(final_summary_df, criteria_dict)
                        
                        # Apply styling to the report for better readability
                        report_subset_cols = [col for col in validation_report.columns if 'Check' in col or 'Status' in col]
                        st.dataframe(
                            validation_report.style.apply(lambda col: col.map(style_report), subset=report_subset_cols),
                            use_container_width=True
                        )


                    # --- 4. Create separate boxplots for each level ---
                    st.header("Comparative Boxplots by Level")

                    # Create a clean copy for plotting to avoid data type issues
                    plot_df = df.copy()
                    # Ensure compound and Level columns are numeric
                    for compound in selected_compounds:
                        plot_df[compound] = pd.to_numeric(plot_df[compound], errors='coerce')
                    plot_df['Level'] = pd.to_numeric(plot_df['Level'], errors='coerce')
                    plot_df.dropna(subset=['Level'], inplace=True)
                    
                    if plot_df.empty:
                        st.warning("No data available to generate plots for the selected compounds.")
                    else:
                        # Get sorted unique levels to display plots in an orderly fashion
                        unique_levels = sorted(plot_df['Level'].unique())

                        if not unique_levels:
                            st.warning("Could not find any valid numeric levels in the data.")
                        else:
                            # Loop through each level and create a separate plot
                            for level in unique_levels:
                                # Filter the dataframe for the current level
                                df_for_level = plot_df[plot_df['Level'] == level]
                                
                                # Generate and display the plot
                                level_fig = create_level_specific_boxplot(df_for_level, selected_compounds, level)
                                
                                if level_fig:
                                    st.plotly_chart(level_fig, use_container_width=True)
                                else:
                                    st.warning(f"Could not generate a boxplot for Level {level}. There might be no valid data for the selected compounds at this level.")


    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
        st.info("Please ensure your file is a valid Excel file with a sheet named 'Validation data'.")

else:
    st.info("Awaiting for an Excel file to be uploaded.")




