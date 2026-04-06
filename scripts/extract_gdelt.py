from google.cloud import bigquery
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.entity_resolver import EntityResolver
import os

def read_sql_file(filepath):
    with open(filepath, 'r') as f:
        return f.read()

def execute_query_to_gcs(client, query, output_uri):
    export_query = f"""
    EXPORT DATA OPTIONS(
      uri='{output_uri}',
      format='CSV',
      overwrite=true,
      header=true
    ) AS
    {query}
    """
    
    job = client.query(export_query)
    result = job.result()
    return result

def build_tone_query(template, companies, entity_resolver, start_date, end_date):
    """Build tone extraction query with entity resolution for all companies"""
    
    company_cases = []
    company_conditions = []
    
    for company_name, ticker in companies:
        company_regex = entity_resolver.get_regex_pattern(company_name, ticker)
        company_cases.append(
            f"      WHEN REGEXP_CONTAINS(V2Organizations, r'(?i)\\b({company_regex})\\b') THEN '{company_name}'"
        )
        company_conditions.append(
            f"REGEXP_CONTAINS(V2Organizations, r'(?i)\\b({company_regex})\\b')"
        )
    
    case_clause = '\n'.join(company_cases)
    where_clause = ' OR '.join(company_conditions)
    
    query = f"""
SELECT 
  DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
  CASE
{case_clause}
  END as company,
  COUNT(DISTINCT DocumentIdentifier) as daily_exposure_count,
  AVG(CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64)) as daily_avg_tone
FROM 
  `gdelt-bq.gdeltv2.gkg_partitioned`
WHERE 
  _PARTITIONDATE BETWEEN '{start_date}' AND '{end_date}'
  AND ({where_clause})
GROUP BY event_date, company
ORDER BY event_date ASC
"""
    return query
 
def build_themes_query(companies, entity_resolver, start_date, end_date):
    """Build themes extraction query with entity resolution"""
    
    case_statements = []
    where_conditions = []
    
    for company_name, ticker in companies:
        company_regex = entity_resolver.get_regex_pattern(company_name, ticker)
        case_statements.append(
            f"      WHEN REGEXP_CONTAINS(V2Organizations, r'(?i)\\b({company_regex})\\b') THEN '{company_name}'"
        )
        where_conditions.append(
            f"      REGEXP_CONTAINS(V2Organizations, r'(?i)\\b({company_regex})\\b')"
        )
    
    case_clause = '\n'.join(case_statements)
    where_clause = ' OR\n'.join(where_conditions)
    
    query = f"""
WITH filtered AS (
  -- Filter the raw data down to just your target companies
  SELECT 
    DATE(PARSE_TIMESTAMP('%Y%m%d%H%M%S', CAST(DATE AS STRING))) as event_date,
    V2Themes,
    CAST(SPLIT(V2Tone, ',')[OFFSET(0)] AS FLOAT64) as primary_tone,
    DocumentIdentifier,
    CASE 
{case_clause}
    END as company
  FROM 
    `gdelt-bq.gdeltv2.gkg_partitioned`
  WHERE 
    _PARTITIONDATE BETWEEN '{start_date}' AND '{end_date}'
    AND V2Themes IS NOT NULL
    AND (
{where_clause}
    )
)
 
-- Unnest the themes and aggregate by day, company, and theme
SELECT 
  event_date,
  company,
  SPLIT(SPLIT(individual_theme, '_')[OFFSET(0)], ',')[OFFSET(0)] as theme_category,
  COUNT(DISTINCT DocumentIdentifier) as daily_theme_mentions,
  AVG(primary_tone) as daily_theme_avg_tone
FROM 
  filtered,
  UNNEST(SPLIT(V2Themes, ';')) as individual_theme
WHERE 
  individual_theme != ''
GROUP BY 
  event_date,
  company,
  theme_category
ORDER BY 
  event_date DESC,
  company,
  theme_category
"""
    return query
 
def main():
    client = bigquery.Client(project='gdelt-stock-sentiment-analysis')
    
    entity_resolver = EntityResolver()
    
    # Define companies to extract
    companies = [
        ('Amazon', 'AMZN'),
        ('Pfizer', 'PFE'),
        ('Aramco', '2222.SR')
    ]
    
    # Date range
    start_date = '2020-01-01'
    end_date = '2025-12-31'
    
    sql_dir = Path(__file__).parent.parent / 'sql'
    gcs_bucket = 'gs://og-gdelt-main-data-dev'
    
    # Extract tone data with entity resolution
    tone_template = read_sql_file(sql_dir / 'tone_extract.sql')
    tone_query = build_tone_query(tone_template, companies, entity_resolver, start_date, end_date)

    try:
        execute_query_to_gcs(
            client=client,
            query=tone_query,
            output_uri=f'{gcs_bucket}/gdelt_raw/tone_exposure_data_*.csv'
        )
    except Exception as e:
        print(f"✗ Error executing tone_extract: {e}\n")
    
    # Extract themes data with entity resolution
    themes_query = build_themes_query(companies, entity_resolver, start_date, end_date)
    
    try:
        execute_query_to_gcs(
            client=client,
            query=themes_query,
            output_uri=f'{gcs_bucket}/gdelt_raw/themes_data*.csv'
        )
    except Exception as e:
        print(f"✗ Error executing themes_extract: {e}\n")
    

    for company_name, ticker in companies:
        pattern = entity_resolver.get_regex_pattern(company_name, ticker)
        print(f"{company_name} ({ticker}):")
        print(f"  {pattern}\n")

if __name__ == '__main__':
    main()