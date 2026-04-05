import zipfile
import csv
import pandas as pd

def extract_gkg_columns(zip_file_path, columns_needed):

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        csv_file = None
        for filename in zip_ref.namelist():
            if filename.endswith('.csv'):
                csv_file = filename
                break
        
        if not csv_file:
            raise FileNotFoundError("No CSV file found in the zip archive")
        
        with zip_ref.open(csv_file) as csv_content:
            csv_reader = csv.reader(csv_content.read().decode('utf-8').splitlines(), delimiter='\t')
            
            all_rows = list(csv_reader)
            
            extracted_data = []
            for row in all_rows:
                row_data = {}
                for col_idx, col_name in columns_needed.items():
                    if col_idx <= len(row):
                        row_data[col_name] = row[col_idx - 1]
                    else:
                        row_data[col_name] = None
                extracted_data.append(row_data)
            
            df = pd.DataFrame(extracted_data)
            
            return df


if __name__ == "__main__":
    zip_file_path = "output.zip"
    
    columns_needed = {
        1: "GKGRECORDID",
        2: "DATE",
        9: "V2Themes",
        11: "V2Locations",
        13: "V2Persons",
        15: "V2Organizations",
        16: "V2Tone"
    }
    
    df = extract_gkg_columns(zip_file_path, columns_needed)

