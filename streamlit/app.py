import streamlit as st
from google.cloud import bigquery

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

@st.cache_data
def load_data(company_name: str):
    client = bigquery.Client()

    query = f"""
            SELECT *
            FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean` db
            WHERE db.company = '{company_name}' 
            LIMIT 100
            """
    return client.query(query).to_dataframe()


st.title("GDELT Data Clean Data Preview")

company_selected = st.selectbox(
    "Select a company",
    options=list(COMPANIES.values()),
    index=None,
    placeholder="Select a company to visualize",
)

if company_selected:
    # Optional: Adds a nice loading spinner while BigQuery fetches the data
    with st.spinner(f"Loading data for {company_selected}..."):
        df = load_data(company_selected)
        st.dataframe(df)


# connection with GCP storage
from st_files_connection import FilesConnection


conn = st.connection('gcs', type=FilesConnection)

st.title("Q1")
df2 = conn.read(f"{BUCKETS['q1']}/q1_tone_correlation_results.csv", input_format="csv", ttl=600)

st.title("Q2")
df3 = conn.read(f"{BUCKETS['q2']}/q2_top_themes_amazon.csv", input_format="csv", ttl=600)
st.title("Q3")
df4 = conn.read(f"{BUCKETS['q3']}/q3_exposure_correlation_results.csv", input_format="csv", ttl=600)


