import requests
import re
from typing import List, Optional
from datetime import datetime
import os
import boto3
from botocore.exceptions import ClientError
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from inspect_gkg_csv import extract_gkg_columns

# Column mapping for GKG data extraction
GKG_COLUMNS = {
    1: "GKGRECORDID",
    2: "DATE",
    9: "V2Themes",
    11: "V2Locations",
    13: "V2Persons",
    15: "V2Organizations",
    16: "V2Tone"
}


def create_partitioned_s3_key(timestamp: str, filename: str, s3_prefix: str) -> str:
    """
    Create a partitioned S3 key based on timestamp.
    
    Args:
        timestamp: Timestamp in format 'YYYYMMDDHHMMSS'
        filename: Original filename
        s3_prefix: Base S3 prefix
    
    Returns:
        Partitioned S3 key in format: prefix/year=YYYY/month=MM/day=DD/hour=HH/filename
    """
    year = timestamp[:4]
    month = timestamp[4:6]
    day = timestamp[6:8]
    hour = timestamp[8:10]
    
    partitioned_key = f"{s3_prefix.rstrip('/')}/year={year}/month={month}/day={day}/hour={hour}/{filename}"
    return partitioned_key


def process_single_file(
    url: str,
    idx: int,
    total: int,
    bucket_name: str,
    s3_prefix: str,
    s3_client,
) -> Optional[str]:
    filename = url.split('/')[-1]
    
    # Extract timestamp from filename for partitioning
    match = re.search(r'(\d{14})\.gkg\.csv\.zip', filename)
    if match:
        timestamp = match.group(1)
        parquet_filename = filename.replace('.csv.zip', '.parquet')
        s3_key = create_partitioned_s3_key(timestamp, parquet_filename, s3_prefix)
    else:
        parquet_filename = filename.replace('.csv.zip', '.parquet')
        s3_key = f"{s3_prefix.rstrip('/')}/{parquet_filename}"
    
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        print(f"[{idx}/{total}] Skipping {filename} (already exists in S3)")
        return s3_key
    except ClientError:
        pass
    
    try:
        print(f"[{idx}/{total}] Downloading {filename}...")
        file_response = requests.get(url, stream=True, timeout=60)
        file_response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name
            for chunk in file_response.iter_content(chunk_size=8192):
                tmp_zip.write(chunk)
        
        try:
            print(f"[{idx}/{total}] Extracting and transforming data...")
            df = extract_gkg_columns(tmp_zip_path, GKG_COLUMNS)
            
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp_parquet:
                tmp_parquet_path = tmp_parquet.name
            
            df.to_parquet(tmp_parquet_path, engine='pyarrow', compression='snappy', index=False)
            
            try:
                print(f"[{idx}/{total}] Uploading Parquet to s3://{bucket_name}/{s3_key}...")
                s3_client.upload_file(
                    tmp_parquet_path,
                    bucket_name,
                    s3_key,
                    ExtraArgs={
                        'ContentType': 'application/octet-stream'
                    }
                )
                
                print(f"[{idx}/{total}] ✓ Successfully uploaded {df.shape[0]} rows with {df.shape[1]} columns (Parquet)")
                return s3_key
                
            finally:
                if os.path.exists(tmp_parquet_path):
                    os.unlink(tmp_parquet_path)
        finally:
            if os.path.exists(tmp_zip_path):
                os.unlink(tmp_zip_path)
        
    except Exception as e:
        print(f"[{idx}/{total}] ✗ Error processing {filename}: {e}")
        return None


def crawl_gdelt_gkg_files_to_s3(
    bucket_name: str,
    s3_prefix: str = "gdelt/gkg/",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: Optional[int] = None,
    aws_profile: Optional[str] = None,
    region_name: str = "us-east-1",
    max_workers: int = 5
) -> List[str]:
    
    masterlist_url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
    
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3_client = session.client('s3', region_name=region_name)
    
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Connected to S3 bucket: {bucket_name}")
    except ClientError as e:
        print(f"Error accessing bucket {bucket_name}: {e}")
        raise
    
    print(f"Fetching master file list from {masterlist_url}...")
    with open("masterfilelist.txt", "r") as f:
        lines = f.read().strip().split('\n')
        gkg_urls = []
    
    for line in lines:
        # Each line format: <size> <hash> <url>
        parts = line.split()
        if len(parts) >= 3:
            url = parts[2]
            if 'gkg.csv.zip' in url.lower():
                match = re.search(r'/(\d{14})\.gkg\.csv\.zip', url)
                if match:
                    timestamp = match.group(1)
                    
                    if start_date and timestamp < start_date:
                        continue
                    if end_date and timestamp > end_date:
                        continue
                    
                    gkg_urls.append(url)
    
    print(f"Found {len(gkg_urls)} GKG files")
    
    if limit:
        gkg_urls = gkg_urls[:limit]
        print(f"Limiting to {limit} files")
    
    print(f"\nProcessing files with {max_workers} parallel workers...\n")
    
    # Process files in parallel
    uploaded_keys = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(
                process_single_file,
                url,
                idx,
                len(gkg_urls),
                bucket_name,
                s3_prefix,
                s3_client
            ): url
            for idx, url in enumerate(gkg_urls, 1)
        }
        
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                s3_key = future.result()
                if s3_key:
                    uploaded_keys.append(s3_key)
            except Exception as e:
                filename = url.split('/')[-1]
                print(f"Unexpected error processing {filename}: {e}")
    
    print(f"\n{'='*80}")
    print(f"Upload complete! {len(uploaded_keys)} files uploaded to s3://{bucket_name}/{s3_prefix}")
    print(f"{'='*80}")
    return 0



if __name__ == "__main__":
    uploaded_keys = crawl_gdelt_gkg_files_to_s3(
        bucket_name="og-gdelt-gkg-data-bucket",
        s3_prefix="gdelt/gkg/",
        start_date="20240303000000",
        end_date="20260101000000",
        limit=60000,
        region_name="us-east-1"
    )
    
    print(f"\nUploaded files to S3:")
    for key in uploaded_keys:
        print(f"  - {key}")
