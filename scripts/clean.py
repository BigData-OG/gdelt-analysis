from google.cloud import bigquery

def export_partitioned_by_month(client, table_name, base_uri, file_prefix):
    """
    Export data partitioned by month (year-month format).
    """
    # Get distinct year-months from the data
    year_months_query = f"""
    SELECT DISTINCT FORMAT_DATE('%Y-%m', event_date) as year_month
    FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.{table_name}`
    ORDER BY year_month
    """
    
    year_months = client.query(year_months_query).result()
    
    exported_count = 0
    for row in year_months:
        year_month = row.year_month
        
        # Export each month to separate file
        export_query = f"""
        EXPORT DATA OPTIONS(
          uri='{base_uri}/{file_prefix}_{year_month}.csv',
          format='CSV',
          overwrite=true,
          header=true
        ) AS
        SELECT * 
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.{table_name}`
        WHERE FORMAT_DATE('%Y-%m', event_date) = '{year_month}'
        ORDER BY ticker, event_date
        """
        
        client.query(export_query).result()
        exported_count += 1
    
    print(f"  Total: {exported_count} monthly files exported")
    return exported_count
 
def main():
    client = bigquery.Client(project='gdelt-stock-sentiment-analysis')
    
    clean_tone_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data_clean` AS
    WITH filled_prices AS (
        SELECT 
            event_date,
            company,
            ticker,
            daily_exposure_count,
            daily_avg_tone,
            -- Forward fill stock prices using LAST_VALUE window function
            LAST_VALUE(Open IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Open,
            LAST_VALUE(High IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as High,
            LAST_VALUE(Low IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Low,
            LAST_VALUE(Close IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Close,
            LAST_VALUE(Volume IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Volume
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.combined_data`
    )
    SELECT 
        event_date,
        company,
        ticker,
        daily_exposure_count,
        daily_avg_tone,
        Open,
        High,
        Low,
        Close,
        Volume,
        LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date) as prev_close,
        LEAD(Close) OVER (PARTITION BY ticker ORDER BY event_date) as next_day_close,
        SAFE_DIVIDE(Close - LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date), 
                    LAG(Close) OVER (PARTITION BY ticker ORDER BY event_date)) * 100 as daily_return_pct,
        EXTRACT(DAYOFWEEK FROM event_date) as day_of_week
    FROM filled_prices
    WHERE Close IS NOT NULL
    ORDER BY ticker, event_date
    """
    
    client.query(clean_tone_query).result()

    
    clean_themes_query = """
    CREATE OR REPLACE TABLE `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices_clean` AS
    WITH filled_prices AS (
        SELECT 
            event_date,
            company,
            ticker,
            theme_category,
            daily_theme_mentions,
            daily_theme_avg_tone,
            LAST_VALUE(Open IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Open,
            LAST_VALUE(High IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as High,
            LAST_VALUE(Low IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Low,
            LAST_VALUE(Close IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Close,
            LAST_VALUE(Volume IGNORE NULLS) OVER (
                PARTITION BY ticker ORDER BY event_date 
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) as Volume
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.themes_with_prices`
    ),
    next_day_prices AS (
        SELECT
            Ticker as ticker,
            Date as event_date,
            LEAD(Close) OVER (PARTITION BY Ticker ORDER BY Date) as next_day_close
        FROM `gdelt-stock-sentiment-analysis.gdelt_analysis.stock_prices`
    )
    SELECT 
        fp.event_date,
        fp.company,
        fp.ticker,
        fp.theme_category,
        fp.daily_theme_mentions,
        fp.daily_theme_avg_tone,
        fp.Open,
        fp.High,
        fp.Low,
        fp.Close,
        fp.Volume,
        ndp.next_day_close,
        LAG(fp.Close) OVER (PARTITION BY fp.ticker ORDER BY fp.event_date) as prev_close,
        SAFE_DIVIDE(fp.Close - LAG(fp.Close) OVER (PARTITION BY fp.ticker ORDER BY fp.event_date), 
                    LAG(fp.Close) OVER (PARTITION BY fp.ticker ORDER BY fp.event_date)) * 100 as daily_return_pct,
        EXTRACT(DAYOFWEEK FROM fp.event_date) as day_of_week
    FROM filled_prices fp
    LEFT JOIN next_day_prices ndp
        ON fp.ticker = ndp.ticker 
        AND fp.event_date = ndp.event_date
    WHERE fp.Close IS NOT NULL
    ORDER BY fp.ticker, fp.event_date
    """
    
    client.query(clean_themes_query).result()
    
    # Export cleaned data
    export_partitioned_by_month(
        client=client,
        table_name='combined_data_clean',
        base_uri='gs://og-gdelt-main-data-dev/cleaned_data/monthly',
        file_prefix='combined_data_clean'
    )
 
    export_partitioned_by_month(
        client=client,
        table_name='themes_with_prices_clean',
        base_uri='gs://og-gdelt-main-data-dev/cleaned_data/monthly',
        file_prefix='themes_with_prices_clean'
    )
    
    
 
if __name__ == '__main__':
    main()