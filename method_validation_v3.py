import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import io
from openpyxl import Workbook

# --- Page Configuration ---
st.set_page_config(
    page_title="AquOmixLab - Method Validation Report",
    page_icon="🔬",
    layout="wide"
)

# --- Analysis Functions ---
def calculate_summary_stats(df, compound):
    """
    Calculates and returns summary statistics for a given compound, grouped by Level.
    """
    df[compound] = pd.to_numeric(df[compound], errors='coerce')
    df_clean = df.dropna(subset=[compound, 'Level']).copy()
    df_clean['Level'] = pd.to_numeric(df_clean['Level']).astype(float)

    if df_clean.empty:
        return pd.DataFrame()

    summary = df_clean.groupby('Level').agg(
        Mean=(compound, 'mean'), SD=(compound, 'std'),
        Min=(compound, 'min'), Max=(compound, 'max'), N=(compound, 'count')
    ).reset_index()

    summary['%RSD'] = 100 * (summary['SD'] / summary['Mean']).replace([np.inf, -np.inf], None)
    summary['Mean % Recovery'] = 100 * (summary['Mean'] / summary['Level']).replace([np.inf, -np.inf], None)
    
    bias_sq = (summary['Mean'] - summary['Level'])**2
    sd_sq = summary['SD']**2
    summary['Uexp (k=2)'] = 2 * np.sqrt(sd_sq + bias_sq)
    summary['%Uexp (k=2)'] = (100 * (summary['Uexp (k=2)'] / summary['Level'])).replace([np.inf, -np.inf], None)
    
    return summary

def create_level_specific_boxplot(df_level, compounds, level_value):
    """
    Creates a boxplot for a specific level, comparing all selected compounds.
    """
    id_vars = ['Level', 'Analyst', 'Date', 'Sample']
    id_vars_present = [col for col in id_vars if col in df_level.columns]
    
    long_df = df_level.melt(
        id_vars=id_vars_present, value_vars=compounds,
        var_name='Compound', value_name='Measurement'
    )
    long_df['Measurement'] = pd.to_numeric(long_df['Measurement'], errors='coerce')
    long_df['Level'] = pd.to_numeric(long_df['Level'], errors='coerce')
    long_df = long_df.dropna(subset=['Measurement', 'Level'])
    long_df['% Recovery'] = 100 * (long_df['Measurement'] / long_df['Level']).replace([np.inf, -np.inf], None)
    long_df = long_df.dropna(subset=['% Recovery'])
    
    if long_df.empty: return None

    fig = px.box(
        long_df, x='Compound', y='% Recovery', color='Compound',
        title=f'% Recovery Distribution at Level {level_value}',
        labels={'Compound': 'Compound', '% Recovery': '% Recovery'},
        points="all", hover_data=['Analyst', 'Date', 'Sample']
    )
    fig.update_layout(boxgroupgap=0.5, boxgap=0.3)
    return fig

def generate_validation_report(summary_df, criteria_lookup):
    """
    Compares summary statistics against compound-and-level-specific validation criteria.
    """
    report_df = summary_df[['Compound', 'Level']].copy()

    def check(row, criterion_key, value_col):
        lookup_key = (row['Compound'], row['Level'], criterion_key)
        limit = criteria_lookup.get(lookup_key)
        value = row[value_col]
        
        if limit is None or pd.isna(limit): return "N/A"
        if pd.isna(value): return "FAIL"

        if 'min' in criterion_key.lower():
            return "PASS" if value >= limit else "FAIL"
        else:
            return "PASS" if value <= limit else "FAIL"

    report_df['RSD Check'] = summary_df.apply(check, axis=1, criterion_key='%RSD max', value_col='%RSD')
    report_df['Recovery Max Check'] = summary_df.apply(check, axis=1, criterion_key='Mean % Recovery max', value_col='Mean % Recovery')
    report_df['Recovery Min Check'] = summary_df.apply(check, axis=1, criterion_key='Mean % Recovery min', value_col='Mean % Recovery')
    report_df['%Uexp Check'] = summary_df.apply(check, axis=1, criterion_key='%Uexp (k=2) max', value_col='%Uexp (k=2)')

    check_cols = [col for col in report_df.columns if 'Check' in col]
    report_df['Overall Status'] = report_df.apply(lambda row: "FAIL" if "FAIL" in row[check_cols].values else "PASS", axis=1)
    return report_df

def style_report(val):
    color = {'PASS': 'green', 'FAIL': 'red'}.get(val, 'grey')
    return f'color: {color}; font-weight: bold;'

def to_excel(info_df, summary_df, criteria_df, validation_df):
    """
    Creates an in-memory Excel file with multiple sheets for the report.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if info_df is not None:
            info_df.to_excel(writer, sheet_name='Project_Info', index=False)
        if summary_df is not None:
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        if criteria_df is not None:
            criteria_df.to_excel(writer, sheet_name='Applied_Criteria', index=False)
        if validation_df is not None:
            validation_df.to_excel(writer, sheet_name='Validation_Evaluation', index=False)
    processed_data = output.getvalue()
    return processed_data


# --- Main Application UI ---
st.title("AquOmixLab - Method Validation Report")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose your Excel file (Max 100MB)", type="xlsx",
    help="Upload the Excel file with 'Info', 'Validation data', and 'Criteria' sheets."
)

if uploaded_file is not None:
    try:
        # --- Data Loading ---
        df = pd.read_excel(uploaded_file, sheet_name="Validation data")
        
        info_df, criteria_df, criteria_lookup = None, None, {}
        try:
            info_df = pd.read_excel(uploaded_file, sheet_name="Info", header=None).dropna(how='all')
        except Exception: st.warning("Optional 'Info' sheet not found.")
        try:
            criteria_df_raw = pd.read_excel(uploaded_file, sheet_name="Criteria", header=0)
            criteria_df = criteria_df_raw.dropna(how='all').reset_index(drop=True)
            
            if len(criteria_df.columns) >= 2:
                criteria_df.rename(columns={
                    criteria_df.columns[0]: 'Criterion',
                    criteria_df.columns[1]: 'Level'
                }, inplace=True)
                
                criteria_df['Criterion'] = criteria_df['Criterion'].str.strip()
                criteria_df['Level'] = pd.to_numeric(criteria_df['Level'], errors='coerce').astype(float)
                criteria_df.dropna(subset=['Level'], inplace=True)

                melted_criteria = criteria_df.melt(
                    id_vars=['Criterion', 'Level'], var_name='Compound', value_name='Value'
                ).dropna(subset=['Value'])
                
                melted_criteria['Compound'] = melted_criteria['Compound'].str.strip()
                melted_criteria['Value'] = pd.to_numeric(melted_criteria['Value'], errors='coerce')

                criteria_lookup = melted_criteria.set_index(['Compound', 'Level', 'Criterion'])['Value'].to_dict()
                
                st.sidebar.success("Successfully loaded 'Criteria' sheet.")
            else:
                st.sidebar.warning("'Criteria' sheet is improperly formatted.")
                criteria_df = None
        except Exception as e: 
            st.sidebar.warning(f"Could not process 'Criteria' sheet: {e}.")
            criteria_lookup = {}
            criteria_df = None

        # --- Data Validation ---
        required_cols = ["Index", "Date", "Analyst", "Sample", "Units", "Level", "Notes"]
        cols_in_file = {c.strip().lower(): c for c in df.columns}
        missing_cols, rename_map = [], {}
        for req_col in required_cols:
            if req_col.lower() in cols_in_file:
                rename_map[cols_in_file[req_col.lower()]] = req_col
            else: missing_cols.append(req_col)

        if missing_cols:
            st.error(f"Error: Missing required columns: **{', '.join(missing_cols)}**")
        else:
            df.rename(columns=rename_map, inplace=True)
            df['Level'] = pd.to_numeric(df['Level'], errors='coerce').astype(float) 
            notes_index = df.columns.get_loc("Notes")
            compound_columns_raw = df.columns[notes_index + 1:].tolist()
            compound_columns = [c.strip() for c in compound_columns_raw]
            df.columns = list(df.columns[:notes_index+1]) + compound_columns

            if not compound_columns:
                st.warning("No compound data columns found after the 'Notes' column.")
            else:
                st.success(f"File uploaded successfully! Found {len(compound_columns)} compound(s).")
                
                st.sidebar.header("Analysis Options")
                selected_compounds = st.sidebar.multiselect(
                    "Select compounds to analyze:",
                    options=compound_columns, default=compound_columns
                )

                if not selected_compounds:
                    st.info("Please select at least one compound to start.")
                else:
                    if info_df is not None and not info_df.empty:
                        st.subheader("Project Information")
                        st.table(info_df.rename(columns={0: "Property", 1: "Value"}))
                        st.markdown("---")

                    st.header("Summary Results")
                    with st.expander("View Summary Statistics", expanded=True):
                        all_summaries = []
                        for compound in selected_compounds:
                            summary_df = calculate_summary_stats(df.copy(), compound)
                            if not summary_df.empty:
                                summary_df['Compound'] = compound
                                all_summaries.append(summary_df)
                        
                        if not all_summaries:
                            st.warning("No valid data to calculate statistics.")
                        else:
                            final_summary_df = pd.concat(all_summaries, ignore_index=True)
                            cols_order = ['Compound', 'Level', 'Mean', 'SD', 'Min', 'Max', 'N', '%RSD', 'Mean % Recovery', 'Uexp (k=2)', '%Uexp (k=2)']
                            final_summary_df = final_summary_df[cols_order]
                            st.dataframe(final_summary_df.style.format({
                                'Mean': '{:.3f}', 'SD': '{:.3f}', 'Min': '{:.3f}', 'Max': '{:.3f}', 'N': '{}',
                                '%RSD': '{:.2f}%', 'Mean % Recovery': '{:.2f}%',
                                'Uexp (k=2)': '{:.3f}', '%Uexp (k=2)': '{:.2f}%'
                            }), use_container_width=True)
                    
                    st.header("Comparative Boxplots by Level")
                    plot_df = df.copy()
                    
                    if plot_df.empty or 'Level' not in plot_df or plot_df['Level'].isnull().all():
                        st.warning("No data to generate plots.")
                    else:
                        unique_levels = sorted(plot_df['Level'].dropna().unique())
                        for level in unique_levels:
                            df_for_level = plot_df[plot_df['Level'] == level]
                            level_fig = create_level_specific_boxplot(df_for_level, selected_compounds, level)
                            if level_fig:
                                st.plotly_chart(level_fig, use_container_width=True)
                    
                    with st.expander("View Uncertainty Calculation Details"):
                        st.markdown("""
                        The **Expanded Uncertainty (`Uexp`)** provides an interval where the true value is believed to lie with a 95% confidence level.
                        - The calculation is based on the **NORDTEST TR 537** methodology.
                        - **Calculation**: `Uexp = 2 * sqrt(SD² + Bias²)`, where `Bias = Mean - Level`.
                        """)

                    if criteria_lookup is not None and 'final_summary_df' in locals():
                        st.header("Validation Evaluation")
                        if criteria_df is not None:
                            st.subheader("Applied Validation Criteria")
                            st.dataframe(criteria_df.style.format({'Level': '{:.0f}'}))
                        
                        validation_report = generate_validation_report(final_summary_df, criteria_lookup)
                        report_subset_cols = [col for col in validation_report.columns if 'Check' in col or 'Status' in col]
                        st.subheader("PASS/FAIL Evaluation")
                        st.dataframe(
                            validation_report.style.format({'Level': '{:.0f}'}).apply(lambda col: col.map(style_report), subset=report_subset_cols),
                            use_container_width=True
                        )

                    # --- Excel Download Section ---
                    st.markdown("---")
                    st.header("Download Report")
                    
                    if 'final_summary_df' in locals() and 'validation_report' in locals():
                        excel_data = to_excel(info_df, final_summary_df, criteria_df, validation_report)
                        st.download_button(
                            label="📥 Download Report (Excel)",
                            data=excel_data,
                            file_name="validation_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    else:
                        st.info("Process data first to enable the download button.")


    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.subheader("How it Works")
    st.markdown("""
    This application performs a comprehensive analysis of analytical method validation data. Simply upload your Excel file to get started.
    
    **App Features:**
    - **Summary Statistics:** Calculates key metrics like Mean, Standard Deviation (SD), %RSD, and Min/Max values for each compound at each concentration level.
    - **Recovery & Uncertainty:** Computes Mean % Recovery and Expanded Uncertainty (`%Uexp`) based on the NORDTEST TR 537 methodology.
    - **Automated Evaluation:** Compares your results against predefined validation criteria to generate an automated PASS/FAIL report.
    - **Interactive Visualizations:** Generates comparative boxplots to visualize the % Recovery distribution for your compounds.
    """)

    st.subheader("Excel File Structure Requirements")
    st.markdown("""
    For the app to work correctly, your Excel file must contain the following sheets with the specified structure:
    
    1.  **`Info` Sheet (Optional):**
        - Contains general project information.
        - **Column A:** Property name (e.g., "Project Name", "Method ID").
        - **Column B:** Corresponding value.

    2.  **`Validation data` Sheet (Required):**
        - Contains the raw measurement data.
        - **Columns A-G:** `Index`, `Date`, `Analyst`, `Sample`, `Units`, `Level`, `Notes`.
        - **Column H onwards:** Measurement data for each compound. The compound name must be in the first row (the header).

    3.  **`Criteria` Sheet (Optional):**
        - Defines the PASS/FAIL criteria for the validation evaluation.
        - **Column A:** The criterion name (must be one of `'%RSD max'`, `'Mean % Recovery max'`, `'Mean % Recovery min'`, `'%Uexp (k=2) max'`).
        - **Column B:** The concentration `Level` the criterion applies to.
        - **Column C onwards:** The criterion value for each compound. The compound name must be in the header.
    """)
    st.info("Please upload your Excel file above to begin the analysis.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.image("https://raw.githubusercontent.com/trikaloudis/method_validation-/main/Aquomixlab%20Logo%20v2.png", use_container_width=True)
st.sidebar.markdown("[www.aquomixlab.com](https://www.aquomixlab.com)")

