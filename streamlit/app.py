import streamlit as st
from google.cloud import bigquery
from st_files_connection import FilesConnection

# --- Constants & Configuration ---
COMPANIES = {
    'AMZN': 'Amazon',
    '2222.SR': 'Aramco',
    'PFE': 'Pfizer'
}

BUCKETS = {
    'q1': 'og-gdelt-main-data-dev/analysis_results/q1',
    'q2': 'og-gdelt-main-data-dev/analysis_results/q2',
    'q3': 'og-gdelt-main-data-dev/analysis_results/q3'
}

# --- Data Loading Functions ---
@st.cache_data
def load_company_data(company_name: str):
    """Loads a preview of combined clean data for a given company."""
    client = bigquery.Client()

    query = f"""
            SELECT *
            FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean` db
            WHERE db.company = '{company_name}' 
            LIMIT 100
            """
    return client.query(query).to_dataframe()

# --- Main Streamlit App ---
def main():
    # 1. Clean Data Preview Section
    st.title("GDELT Clean Data Preview")

    selected_company = st.selectbox(
        "Select a company",
        options=list(COMPANIES.values()),
        index=None,
        placeholder="Select a company to visualize",
    )

    if selected_company:
        with st.spinner(f"Loading data for {selected_company}..."):
            company_data_df = load_company_data(selected_company)
            st.dataframe(company_data_df)

    # 2. GCP Storage Results Section
    gcs_connection = st.connection('gcs', type=FilesConnection)

    # Question 1 Results
    st.title("Q1: Tone Correlation")
    q1_tone_correlation_df = gcs_connection.read(f"{BUCKETS['q1']}/q1_tone_correlation_results.csv", input_format="csv")
    st.dataframe(q1_tone_correlation_df)

    # Question 2 Results
    st.title("Q2: Top Themes")
    
    st.subheader("Amazon")
    q2_amazon_themes_df = gcs_connection.read(f"{BUCKETS['q2']}/q2_top_themes_amazon.csv", input_format="csv")
    st.dataframe(q2_amazon_themes_df)
    
    st.subheader("Aramco")
    q2_aramco_themes_df = gcs_connection.read(f"{BUCKETS['q2']}/q2_top_themes_aramco.csv", input_format="csv")
    st.dataframe(q2_aramco_themes_df)
    
    st.subheader("Pfizer")
    q2_pfizer_themes_df = gcs_connection.read(f"{BUCKETS['q2']}/q2_top_themes_pfizer.csv", input_format="csv")
    st.dataframe(q2_pfizer_themes_df)

    # Question 3 Results
    st.title("Q3: Exposure Correlation")
    q3_exposure_correlation_df = gcs_connection.read(f"{BUCKETS['q3']}/q3_exposure_correlation_results.csv", input_format="csv")
    st.dataframe(q3_exposure_correlation_df)

if __name__ == "__main__":
    main()
