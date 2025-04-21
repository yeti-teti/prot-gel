# Sample parquet data

import os
import sys
import time

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs
from dotenv import load_dotenv

#  Configuration 
# Cloudflare R2 Configuration
ENV_FILE_PATH = ".env"
R2_BUCKET_NAME = os.getenv("CLOUDFARE_BUCKET_NAME") # Load from env 
# Path WITHIN the R2 bucket where the partitioned dataset resides
R2_DATASET_PATH_IN_BUCKET = "integrated_data/viridiplantae_dataset_partitioned_from_json"
# Sample size
SAMPLE_SIZE = 10

#  Load Environment Variables for R2 
print(f"Loading R2 credentials from: {ENV_FILE_PATH}")
if not load_dotenv(dotenv_path=ENV_FILE_PATH):
    print(f"Warning: .env file not found at {ENV_FILE_PATH}. Using system environment variables.")
    if not all(os.getenv(k) for k in ["CLOUDFARE_ACCESS_KEY", "CLOUDFARE_SECRET_KEY"]) or \
       not R2_BUCKET_NAME or \
       not (os.getenv("CLOUDFARE_ACCOUNT_ID") or os.getenv("CLOUDFARE_ENDPOINT")):
        print("Warning: Critical R2 environment variables might be missing.")

r2_access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
r2_secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
r2_account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")

if not R2_BUCKET_NAME:
     R2_BUCKET_NAME = os.getenv("CLOUDFARE_BUCKET_NAME")
r2_endpoint = os.getenv("CLOUDFARE_ENDPOINT")


# Endpoint from account ID if endpoint not explicitly set
if not r2_endpoint and r2_account_id:
    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"
elif not r2_endpoint and not r2_account_id:
      print("ERROR: Missing Cloudflare R2 endpoint or account ID (set CLOUDFARE_ENDPOINT or CLOUDFARE_ACCOUNT_ID).")
      sys.exit(1)

# Validate required R2 variables
if not all([r2_access_key, r2_secret_key, R2_BUCKET_NAME, r2_endpoint]):
    print("ERROR: Missing Cloudflare R2 credentials/config (KEY, SECRET, BUCKET, ENDPOINT/ACCOUNT_ID) in environment/.env or system environment.")
    sys.exit(1)

# Full R2 Path
FULL_R2_PATH = f"{R2_BUCKET_NAME}/{R2_DATASET_PATH_IN_BUCKET}"

print(f"Target R2 Endpoint: {r2_endpoint}")
print(f"Target R2 Bucket:   {R2_BUCKET_NAME}")
print(f"Target R2 Dataset Path: {R2_DATASET_PATH_IN_BUCKET}")
print(f"Using Full Path for PyArrow Calls: {FULL_R2_PATH}")
print(f"Fetching sample size: {SAMPLE_SIZE}")

#  Main Execution 
def main():
    print("\n Starting R2 Parquet Sample Read Script ")
    start_time = time.time()

    # Configure R2 Filesystem Connection
    print("Configuring R2 filesystem connection...")
    try:
        r2_fs = pafs.S3FileSystem(
            endpoint_override=r2_endpoint,
            access_key=r2_access_key,
            secret_key=r2_secret_key,
            scheme="https"
        )
        # Test connection by checking the target dataset directory info
        print(f"Testing connection to dataset path: {FULL_R2_PATH}")
        file_info = r2_fs.get_file_info(FULL_R2_PATH)
        if file_info.type == pafs.FileType.NotFound:
            print(f"ERROR: Dataset path not found on R2: {FULL_R2_PATH}")
            sys.exit(1)
        else:
            print(f"R2 Connection successful. Path exists (Type: {file_info.type}).")
        print("R2 Connection successful and dataset path exists.")

    except Exception as e:
        print(f"ERROR: Failed to configure or test R2 filesystem connection: {e}")
        sys.exit(1)

    # Read Sample Data from Partitioned Parquet Dataset
    print(f"\nReading sample data (first {SAMPLE_SIZE} rows) from partitioned dataset...")
    try:
        # Use pyarrow.dataset to read partitioned data
        # Pass the path *within* the bucket and the filesystem object
        # 'hive' partitioning assumes directory structure like 'partition_key=value'
        parquet_dataset = ds.dataset(
            FULL_R2_PATH,
            filesystem=r2_fs,
            partitioning="hive" # Crucial for reading partitioned data correctly
        )

        # Efficiently read only the first N rows across partitions
        sample_table = parquet_dataset.head(SAMPLE_SIZE)

        # Convert the Arrow Table sample to a Pandas DataFrame for easy display
        sample_df = sample_table.to_pandas()

        read_time = time.time()
        print(f"Successfully read sample data in {read_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"ERROR: Failed to read Parquet dataset from R2: {e}")
        # Common issues: incorrect path, partitioning scheme, permissions, credentials
        sys.exit(1)

    # Display Sample Data
    print("\n Sample Data ")
    if not sample_df.empty:
        print(f"Shape of sample data: {sample_df.shape}")
        # Check if the partitioning column was included automatically
        # (pyarrow dataset usually includes it when reading)
        if 'uniprot_id_prefix' in sample_df.columns:
             print("Partitioning column 'uniprot_id_prefix' is present.")
        else:
             print("Warning: Partitioning column 'uniprot_id_prefix' seems missing in the read data.")

        print("\nFirst few rows:")
        # Display options for potentially wide dataframes
        pd.set_option('display.max_columns', None) # Show all columns
        pd.set_option('display.width', 1000)      # Adjust display width
        print(sample_df)

        # print("\nDataFrame Info:")
        # sample_df.info() # Uncomment for data types and non-null counts
    else:
        print("No data was read or the dataset sample is empty.")

    # 4. Final Summary
    end_time = time.time()
    print("\n Script Finished ")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print("--")

if __name__ == "__main__":
    main()