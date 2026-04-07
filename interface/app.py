import streamlit as st
import plotly.express as px
from google.cloud import bigquery
from st_files_connection import FilesConnection
import pandas as pd

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
    'q3': 'og-gdelt-main-data-dev/analysis_results/q3',
    'monthly': 'og-gdelt-main-data-dev/cleaned_data/monthly'
}

GREEN = '#00CC96'
RED = '#EF553B'


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


# --- Chart Generation Functions ---
def create_tone_vs_close_chart(dataframe):
    df = dataframe.copy()
    df["correlation_type"] = ["Positive" if r > 0 else "Negative" for r in df["tone_vs_close_r"]]

    fig = px.bar(
        df,
        x="company",
        y="tone_vs_close_r",
        color="correlation_type",
        color_discrete_map={"Positive": GREEN, "Negative": RED},
        title="Tone vs Next-Day Close",
        labels={"tone_vs_close_r": "Pearson r", "company": ""}
    )

    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_yaxes(range=[-0.5, 0.5])

    for index, row in df.iterrows():
        if row.get("tone_vs_close_significant") == "Yes":
            offset = 0.05 if row["tone_vs_close_r"] >= 0 else -0.05
            fig.add_annotation(
                x=row["company"],
                y=row["tone_vs_close_r"] + offset,
                text="<b>*</b>",
                showarrow=False,
                font=dict(size=18)
            )
    return fig


def create_tone_vs_return_chart(dataframe):
    df = dataframe.copy()
    df["correlation_type"] = ["Positive" if r > 0 else "Negative" for r in df["tone_vs_return_r"]]

    fig = px.bar(
        df,
        x="company",
        y="tone_vs_return_r",
        color="correlation_type",
        color_discrete_map={"Positive": GREEN, "Negative": RED},
        title="Tone vs Daily Return %",
        labels={"tone_vs_return_r": "Pearson r", "company": ""}
    )

    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_yaxes(range=[-0.5, 0.5])

    for index, row in df.iterrows():
        if row.get("tone_vs_return_significant") == "Yes":
            offset = 0.05 if row["tone_vs_return_r"] >= 0 else -0.05
            fig.add_annotation(
                x=row["company"],
                y=row["tone_vs_return_r"] + offset,
                text="<b>*</b>",
                showarrow=False,
                font=dict(size=18)
            )
    return fig


def create_top_themes_chart(dataframe, title):
    df = dataframe.copy()
    top_10 = df.sort_values('theme_vs_close_r', ascending=False).head(10)
    fig = px.bar(
        top_10,
        x='theme_vs_close_r',
        y='theme_category',
        orientation='h',
        title=title
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def create_exposure_chart(dataframe):
    df = dataframe.copy()
    fig = px.bar(
        df,
        x='company',
        y='exposure_vs_close_r',
        color='exposure_vs_close_significant',
        color_discrete_map={"Yes": GREEN, "No": RED},
        title="Media Exposure Correlation with Close Price"
    )
    return fig


def create_comparative_themes_chart(top_themes_dict):
    # 1. Combine the dictionary into a single DataFrame
    combined_df = pd.DataFrame()
    for company, df in top_themes_dict.items():
        temp_df = df.copy()
        temp_df['company'] = company
        # Sort values so they display correctly in the horizontal bar chart
        temp_df = temp_df.sort_values("theme_vs_return_r", ascending=True)
        combined_df = pd.concat([combined_df, temp_df], ignore_index=True)

    # 2. Add columns for correlation color and significance markers
    combined_df["correlation_type"] = [
        "Positive" if r > 0 else "Negative" for r in combined_df["theme_vs_return_r"]
    ]
    combined_df["significance_star"] = [
        "*" if sig == "Yes" else "" for sig in combined_df["theme_vs_return_significant"]
    ]

    # 3. Generate the faceted Plotly Express chart
    fig = px.bar(
        combined_df,
        x="theme_vs_return_r",
        y="theme_category",
        color="correlation_type",
        facet_col="company",  # Creates the side-by-side subplots
        facet_col_spacing=0.1,
        text="significance_star",  # Adds the * automatically
        color_discrete_map={"Positive": GREEN, "Negative": RED},
        orientation='h',
        title="Q2: Top Themes by Correlation with Daily Return (* = p < 0.05)",
        labels={"theme_vs_return_r": "Pearson r", "theme_category": "", "company": "Company"}
    )

    # 4. Clean up the layout to match your Matplotlib styling
    # Place text outside the bars
    # fig.update_traces(textposition='outside', textfont_size=16)

    # Un-link the Y-axes so each company shows its own specific themes
    fig.update_yaxes(matches=None, showticklabels=True)

    # Clean up facet titles (removes the default "company=" prefix)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Add the vertical zero line across all facets
    fig.add_vline(x=0, line_width=1, line_color="black")

    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,  # Pushes the legend below the x-axis
            xanchor="center",
            x=0.5
        )
    )

    fig.update_layout(legend_title_text="")

    return fig


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

        col1, col2 = st.columns([1, 1])

        with st.container():
            with col1:
                st.plotly_chart(create_tone_vs_return_chart(q1_df), use_container_width=True)

            with col2:
                st.plotly_chart(create_tone_vs_close_chart(q1_df), use_container_width=True)

        with st.container():
            st.header("Tone Impact Data")
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

    # --- Q2: Top Themes ---
    with tab2:
        st.header("Q2: Top Media Themes Driving Performance")

        amzn_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_amazon.csv")
        aramco_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_aramco.csv")
        pfe_df = load_gcs_data(f"{BUCKETS['q2']}/q2_top_themes_pfizer.csv")

        top_themes_dict = {
            "Amazon": amzn_df.head(10),
            "Aramco": aramco_df.head(10),
            "Pfizer": pfe_df.head(10)
        }

        st.plotly_chart(create_comparative_themes_chart(top_themes_dict), use_container_width=True)
        st.divider()
        sub_tab_amzn, sub_tab_aramco, sub_tab_pfe = st.tabs(["Amazon", "Aramco", "Pfizer"])

        with sub_tab_amzn:
            st.dataframe(amzn_df, hide_index=True, use_container_width=True)

        with sub_tab_aramco:
            # Graph on top, inside a column
            col1, _ = st.columns([1, 1])
            with col1:
                st.plotly_chart(
                    create_top_themes_chart(aramco_df, "Aramco: Top 10 Themes by Close Price Correlation"),
                    use_container_width=True
                )

            # Table on bottom
            st.dataframe(aramco_df, hide_index=True, use_container_width=True)

        with sub_tab_pfe:
            st.dataframe(pfe_df, hide_index=True, use_container_width=True)

    # --- Q3: Exposure Correlation ---
    with tab3:
        st.header("Q3: Exposure Correlation")
        q3_df = load_gcs_data(f"{BUCKETS['q3']}/q3_exposure_correlation_results.csv")

        # Graph on top, inside a column
        col1, _ = st.columns([1, 1])
        with col1:
            st.plotly_chart(create_exposure_chart(q3_df), use_container_width=True)

        # Table on bottom
        st.dataframe(
            q3_df,
            column_config={
                "exposure_vs_close_r": st.column_config.NumberColumn("Exposure R", format="%.3f"),
                "exposure_vs_close_significant": st.column_config.TextColumn("Significant?")
            },
            hide_index=True,
            use_container_width=True
        )


if __name__ == "__main__":
    main()
