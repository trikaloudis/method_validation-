import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from fpdf import FPDF
import io
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(
    page_title="AquOmixLab - Method Validation Report",
    page_icon="🔬",
    layout="wide"
)

# --- PDF Export Class & Functions ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AquOmixLab - Method Validation Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def dataframe_to_table(self, df, col_widths=None, notes=None):
        self.set_font('Arial', 'B', 9)
        # Header
        if col_widths is None:
            col_widths = [30] * len(df.columns)

        for i, col in enumerate(df.columns):
            # Sanitize text for PDF font compatibility
            clean_col = str(col).replace('μ', 'u')
            self.cell(col_widths[i], 7, clean_col, 1, 0, 'C')
        self.ln()
        # Data
        self.set_font('Arial', '', 9)
        for index, row in df.iterrows():
            for i, item in enumerate(row):
                # Sanitize text for PDF font compatibility
                clean_item = str(item).replace('μ', 'u')
                self.cell(col_widths[i], 6, clean_item, 1, 0, 'L')
            self.ln()
        if notes:
            self.set_font('Arial', 'I', 8)
            # Sanitize text for PDF font compatibility
            clean_notes = str(notes).replace('μ', 'u')
            self.multi_cell(0, 5, clean_notes)


def generate_pdf_report(info_df, summary_df, criteria_df, validation_df, figures):
    """
    Generates a complete PDF report from the analysis results.
    """
    pdf = PDF()
    pdf.add_page()

    # --- 1. Info Section ---
    if info_df is not None and not info_df.empty:
        pdf.chapter_title('1. Project Information')
        pdf.set_font('Arial', '', 10)
        for index, row in info_df.iterrows():
            pdf.set_font('Arial', 'B', 10)
            # Sanitize text for PDF font compatibility
            pdf.cell(80, 6, str(row[0]).replace('μ', 'u'), 0, 0, 'L')
            pdf.set_font('Arial', '', 10)
            # Sanitize text for PDF font compatibility
            pdf.cell(0, 6, str(row[1]).replace('μ', 'u'), 0, 1, 'L')
        pdf.ln(10)

    # --- 2. Summary Statistics ---
    pdf.chapter_title('2. Summary Statistics')
    # Custom column widths for the summary table
    summary_widths = [25, 15, 18, 18, 18, 18, 10, 18, 24, 24, 24]
    pdf.dataframe_to_table(summary_df, col_widths=summary_widths)
    pdf.ln(10)
    
    # --- 3. Boxplots ---
    pdf.add_page()
    pdf.chapter_title('3. Comparative Boxplots by Level')
    for i, fig in enumerate(figures):
        if i > 0 and i % 2 == 0: # Add a new page after every 2 plots
            pdf.add_page()
        
        img_bytes = fig.to_image(format="png", width=800, height=500, scale=1)
        img_stream = io.BytesIO(img_bytes)
        pdf.image(img_stream, x=10, w=190)
        pdf.ln(5)

    # --- 4. Validation Criteria & Evaluation ---
    pdf.add_page()
    pdf.chapter_title('4. Validation Evaluation')
    if criteria_df is not None and not criteria_df.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Applied Validation Criteria", 0, 1, 'L')
        criteria_widths = [60, 30]
        pdf.dataframe_to_table(criteria_df, col_widths=criteria_widths)
        pdf.ln(5)

    if validation_df is not None and not validation_df.empty:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "PASS/FAIL Evaluation", 0, 1, 'L')
        validation_widths = [25, 20, 25, 25, 25, 25, 25]
        pdf.dataframe_to_table(validation_df, col_widths=validation_widths)

    # --- Generate PDF output ---
    return pdf.output(dest='S').encode('latin-1')


# --- Analysis Functions ---
def calculate_summary_stats(df, compound, group_by_cols):
    """
    Calculates and returns summary statistics for a given compound.
    """
    df[compound] = pd.to_numeric(df[compound], errors='coerce')
    df = df.dropna(subset=[compound])

    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(group_by_cols).agg(
        Mean=(compound, 'mean'),
        SD=(compound, 'std'),
        Min=(compound, 'min'),
        Max=(compound, 'max'),
        N=(compound, 'count')
    ).reset_index()

    summary['%RSD'] = 100 * (summary['SD'] / summary['Mean']).replace([float('inf'), -float('inf')], None)
    summary['Mean % Recovery'] = 100 * (summary['Mean'] / summary['Level']).replace([float('inf'), -float('inf')], None)
    
    bias_sq = (summary['Mean'] - summary['Level'])**2
    sd_sq = summary['SD']**2
    summary['Uexp (k=2)'] = 2 * np.sqrt(sd_sq + bias_sq)
    summary['%Uexp (k=2)'] = (100 * (summary['Uexp (k=2)'] / summary['Level'])).replace([float('inf'), -float('inf')], None)

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
    long_df['% Recovery'] = 100 * (long_df['Measurement'] / long_df['Level']).replace([float('inf'), -float('inf')], None)
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

def generate_validation_report(summary_df, criteria):
    """
    Compares summary statistics against validation criteria and generates a PASS/FAIL report.
    """
    report_df = summary_df[['Compound', 'Level']].copy()
    
    rsd_max = criteria.get('%RSD max')
    rec_max = criteria.get('Mean % Recovery max')
    rec_min = a=criteria.get('Mean % Recovery min')
    uexp_max = criteria.get('%Uexp (k=2) max')

    if rsd_max is not None:
        report_df['RSD Check'] = summary_df.apply(lambda row: "PASS" if pd.notna(row['%RSD']) and row['%RSD'] <= rsd_max else "FAIL", axis=1)
    else: report_df['RSD Check'] = "N/A"

    if rec_max is not None:
        report_df['Recovery Max Check'] = summary_df.apply(lambda row: "PASS" if pd.notna(row['Mean % Recovery']) and row['Mean % Recovery'] <= rec_max else "FAIL", axis=1)
    else: report_df['Recovery Max Check'] = "N/A"

    if rec_min is not None:
        report_df['Recovery Min Check'] = summary_df.apply(lambda row: "PASS" if pd.notna(row['Mean % Recovery']) and row['Mean % Recovery'] >= rec_min else "FAIL", axis=1)
    else: report_df['Recovery Min Check'] = "N/A"

    if uexp_max is not None:
        report_df['%Uexp Check'] = summary_df.apply(lambda row: "PASS" if pd.notna(row['%Uexp (k=2)']) and row['%Uexp (k=2)'] <= uexp_max else "FAIL", axis=1)
    else: report_df['%Uexp Check'] = "N/A"

    check_cols = [col for col in report_df.columns if 'Check' in col]
    report_df['Overall Status'] = report_df.apply(lambda row: "FAIL" if "FAIL" in row[check_cols].values else "PASS", axis=1)

    return report_df

def style_report(val):
    """Applies color to PASS/FAIL/N/A cells."""
    color = {'PASS': 'green', 'FAIL': 'red'}.get(val, 'grey')
    return f'color: {color}; font-weight: bold;'

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
        
        info_df, criteria_df, criteria_dict = None, None, None
        try:
            info_df = pd.read_excel(uploaded_file, sheet_name="Info", header=None).dropna(how='all')
        except Exception: st.warning("Optional 'Info' sheet not found.")
        try:
            criteria_df = pd.read_excel(uploaded_file, sheet_name="Criteria", header=None).dropna(how='all')
            criteria_df[0] = criteria_df[0].str.strip()
            criteria_dict = criteria_df.set_index(0)[1].to_dict()
            st.sidebar.success("Successfully loaded 'Criteria' sheet.")
        except Exception: st.sidebar.warning("Optional 'Criteria' sheet not found. Validation Evaluation will not be generated.")

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
            notes_index = df.columns.get_loc("Notes")
            compound_columns = df.columns[notes_index + 1:].tolist()
            if not compound_columns:
                st.warning("No compound data columns found after the 'Notes' column.")
            else:
                st.success(f"File uploaded successfully! Found {len(compound_columns)} compound(s).")
                
                # --- Sidebar for Compound Selection ---
                st.sidebar.header("Analysis Options")
                selected_compounds = st.sidebar.multiselect(
                    "Select compounds to analyze:",
                    options=compound_columns, default=compound_columns
                )

                if not selected_compounds:
                    st.info("Please select at least one compound to start.")
                else:
                    # --- Display Info Sheet ---
                    if info_df is not None and not info_df.empty:
                        st.subheader("Project Information")
                        st.table(info_df.rename(columns={0: "Property", 1: "Value"}))
                        st.markdown("---")

                    # --- 1. Summary Results Section ---
                    st.header("Summary Results")
                    with st.expander("View Summary Statistics", expanded=True):
                        all_summaries, group_by_cols = [], ['Level']
                        for compound in selected_compounds:
                            summary_df = calculate_summary_stats(df, compound, group_by_cols)
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
                    
                    # --- 2. Boxplots Section ---
                    st.header("Comparative Boxplots by Level")
                    plot_df = df.copy()
                    for compound in selected_compounds:
                        plot_df[compound] = pd.to_numeric(plot_df[compound], errors='coerce')
                    plot_df['Level'] = pd.to_numeric(plot_df['Level'], errors='coerce')
                    plot_df.dropna(subset=['Level'], inplace=True)
                    
                    if plot_df.empty:
                        st.warning("No data to generate plots.")
                    else:
                        unique_levels, figures_list = sorted(plot_df['Level'].unique()), []
                        for level in unique_levels:
                            df_for_level = plot_df[plot_df['Level'] == level]
                            level_fig = create_level_specific_boxplot(df_for_level, selected_compounds, level)
                            if level_fig:
                                st.plotly_chart(level_fig, use_container_width=True)
                                figures_list.append(level_fig)
                    
                    # --- 3. Uncertainty Explanation Section ---
                    with st.expander("View Uncertainty Calculation Details"):
                        st.markdown("""
                        The **Expanded Uncertainty (`Uexp`)** provides an interval where the true value is believed to lie with a 95% confidence level.
                        - The calculation is based on the **NORDTEST TR 537** methodology.
                        - It combines random effects (precision) and systematic effects (bias).
                        - **Calculation**: `Uexp = 2 * sqrt(SD² + Bias²)`, where `Bias = Mean - Level`.
                        - **%Uexp** expresses this uncertainty as a percentage of the concentration level.
                        """)

                    # --- 4. Validation Evaluation Section ---
                    if criteria_dict and 'final_summary_df' in locals():
                        st.header("Validation Evaluation")
                        if criteria_df is not None:
                            st.subheader("Applied Validation Criteria")
                            st.table(criteria_df.rename(columns={0: "Criterion", 1: "Value"}))
                        
                        validation_report = generate_validation_report(final_summary_df, criteria_dict)
                        report_subset_cols = [col for col in validation_report.columns if 'Check' in col or 'Status' in col]
                        st.subheader("PASS/FAIL Evaluation")
                        st.dataframe(
                            validation_report.style.apply(lambda col: col.map(style_report), subset=report_subset_cols),
                            use_container_width=True
                        )

                    # --- 5. PDF Export Button ---
                    st.markdown("---")
                    st.header("Export Report")
                    if 'final_summary_df' in locals() and not final_summary_df.empty:
                        pdf_bytes = generate_pdf_report(
                            info_df, 
                            final_summary_df.round(3), 
                            criteria_df, 
                            validation_report, 
                            figures_list
                        )
                        st.download_button(
                           label="📥 Download Full Report (PDF)",
                           data=pdf_bytes,
                           file_name=f"Validation_Report_{datetime.now().strftime('%Y-%m-%d')}.pdf",
                           mime="application/pdf"
                        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Awaiting for an Excel file to be uploaded.")

# --- Sidebar Footer ---
st.sidebar.markdown("---")
st.sidebar.image("AquOmixLogo.png", use_container_width=True)
st.sidebar.markdown("[https://www.aquomixlab.com](https://www.aquomixlab.com)")


