import streamlit as st
import plotly.express as px
from google.cloud import bigquery
from st_files_connection import FilesConnection

# --- Constants & Configuration ---
# Must be the first Streamlit command
st.set_page_config(
    page_title="GDELT Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

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


@st.cache_data
def load_gcs_data(path: str):
    """Fetches CSV data from GCS and caches it."""
    gcs_connection = st.connection('gcs', type=FilesConnection)
    return gcs_connection.read(path, input_format="csv")


# --- Main Streamlit App ---
def main():
    st.title("📊 GDELT Stock Sentiment Analysis")
    st.markdown("Analyze the correlations between global media tone, exposure, themes, and stock performance.")

    # 1. Sidebar for Raw Data Preview
    with st.sidebar:
        st.header("🗃️ Raw Data Preview")
        st.markdown("Preview the cleaned BigQuery dataset.")
        selected_company = st.selectbox(
            "Select a company",
            options=list(COMPANIES.values()),
            index=None,
            placeholder="Select to load data...",
        )

        if selected_company:
            with st.spinner(f"Querying BQ for {selected_company}..."):
                company_data_df = load_company_data(selected_company)
                st.metric("Rows Retrieved", len(company_data_df))
                st.dataframe(company_data_df, use_container_width=True)

    # 2. Main Content Tabs for Results
    tab1, tab2, tab3 = st.tabs(["Tone Correlation (Q1)", "Top Themes (Q2)", "Exposure Correlation (Q3)"])

    # --- Q1: Tone Correlation ---
    with tab1:
        st.header("Q1: Tone vs. Stock Performance")
        q1_df = load_gcs_data(f"{BUCKETS['q1']}/q1_tone_correlation_results.csv")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(
                q1_df,
                column_config={
                    "tone_vs_close_r": st.column_config.NumberColumn("Close R", format="%.3f"),
                    "tone_vs_close_significant": st.column_config.TextColumn("Significant?"),
                    "n_observations": st.column_config.NumberColumn("Obs.")
                },
                hide_index=True,
                use_container_width=True
            )
        with col2:
            fig_q1 = px.bar(
                q1_df,
                x='company',
                y='tone_vs_close_r',
                color='tone_vs_close_significant',
                color_discrete_map={"Yes": "#00CC96", "No": "#EF553B"},
                title="Tone Correlation with Close Price (Pearson r)"
            )
            st.plotly_chart(fig_q1, use_container_width=True)

    # --- Q2: Top Themes ---
    with tab2:
        st.header("Q2: Top Media Themes Driving Performance")
        # Sub-tabs for each company to prevent visual clutter
        sub_tab_amzn, sub_tab_aramco, sub_tab_pfe = st.tabs(["Amazon", "Aramco", "Pfizer"])

        with sub_tab_amzn:
            amzn_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_amazon.csv")
            st.dataframe(amzn_df, hide_index=True, use_container_width=True)

        with sub_tab_aramco:
            aramco_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_aramco.csv")
            col_a, col_b = st.columns([1, 1])
            with col_a:
                st.dataframe(aramco_df, hide_index=True, use_container_width=True)
            with col_b:
                top_10_aramco = aramco_df.sort_values('theme_vs_close_r', ascending=False).head(10)
                fig_aramco = px.bar(
                    top_10_aramco,
                    x='theme_vs_close_r',
                    y='theme_category',
                    orientation='h',
                    title="Aramco: Top 10 Themes by Close Price Correlation"
                )
                fig_aramco.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig_aramco, use_container_width=True)

        with sub_tab_pfe:
            pfe_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_pfizer.csv")
            st.dataframe(pfe_df, hide_index=True, use_container_width=True)

    # --- Q3: Exposure Correlation ---
    with tab3:
        st.header("Q3: Exposure Correlation")
        q3_df = load_gcs_data(f"{BUCKETS['q3']}/q3_exposure_correlation_results.csv")

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.dataframe(
                q3_df,
                column_config={
                    "exposure_vs_close_r": st.column_config.NumberColumn("Exposure R", format="%.3f"),
                    "exposure_vs_close_significant": st.column_config.TextColumn("Significant?")
                },
                hide_index=True,
                use_container_width=True
            )
        with col2:
            fig_q3 = px.bar(
                q3_df,
                x='company',
                y='exposure_vs_close_r',
                color='exposure_vs_close_significant',
                color_discrete_map={"Yes": "#00CC96", "No": "#EF553B"},
                title="Media Exposure Correlation with Close Price"
            )
            st.plotly_chart(fig_q3, use_container_width=True)


if __name__ == "__main__":
    main()
