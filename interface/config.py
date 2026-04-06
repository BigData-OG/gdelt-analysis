import os

def load():
    config = {
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "GCP_BUCKET": os.getenv("GCP_BUCKET"),
        "BIGQUERY_DATASET": os.getenv("BIGQUERY_DATASET"),
    }
    return config