# Extract the .gz alphafold 

import os
import gzip
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
import sys


BASE_DATA_DIR = "../../data"  # Adjust if your base 'data' directory is elsewhere
INPUT_FOLDER_NAME = "alphafold_sp_pdb"
OUTPUT_FOLDER_NAME = "alphafold_sp_pdb_extracted" # Name for the new output folder

# Maximum number of parallel extraction processes (adjust based on your CPU cores and disk speed)
# None will use os.cpu_count() * 5 by default which is often too high for I/O tasks
# Start with the number of CPU cores or slightly more.
MAX_WORKERS = os.cpu_count() or 6 # Use number of CPU cores as a starting point

# --- Path Setup ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_abs_path = os.path.abspath(os.path.join(script_dir, BASE_DATA_DIR))
    input_dir_path = os.path.join(base_abs_path, "structure_databases", INPUT_FOLDER_NAME)
    output_dir_path = os.path.join(base_abs_path, "structure_databases", OUTPUT_FOLDER_NAME)
except Exception as e:
    print(f"Error determining script directory or paths: {e}")
    print("Please ensure the script is placed correctly relative to the BASE_DATA_DIR.")
    sys.exit(1)


# --- Helper Function for Extraction ---
def extract_pdb_gz(gz_filepath, output_dir):
    """
    Extracts a single .pdb.gz file to the output directory.

    Args:
        gz_filepath (str): Full path to the input .pdb.gz file.
        output_dir (str): Full path to the directory where the .pdb file should be saved.

    Returns:
        tuple: (bool, str) indicating success status and a message/error.
    """
    try:
        base_filename = os.path.basename(gz_filepath)
        # Remove the '.gz' extension to get the output filename
        if base_filename.lower().endswith(".pdb.gz"):
            output_filename = base_filename[:-3] # Remove last 3 chars (.gz)
        else:
            # Handle cases where file might not end exactly with .pdb.gz (less common)
            # This basic approach assumes .gz is the only extension to remove
            output_filename, _ = os.path.splitext(base_filename)
            if not output_filename.lower().endswith(".pdb"):
                 output_filename += ".pdb" # Ensure it ends with .pdb

        output_filepath = os.path.join(output_dir, output_filename)

        # Extract using gzip and shutil.copyfileobj for efficiency
        with gzip.open(gz_filepath, 'rb') as f_in:
            with open(output_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        return True, gz_filepath # Return input path on success for tracking

    except FileNotFoundError:
        return False, f"Input file not found: {gz_filepath}"
    except gzip.BadGzipFile:
        return False, f"Bad Gzip file: {gz_filepath}"
    except Exception as e:
        return False, f"Failed to extract {gz_filepath}: {e}"

# --- Main Execution ---
def main():
    print("--- Starting PDB.GZ Extraction Script ---")
    start_time = time.time()

    # 1. Validate Input Directory
    if not os.path.isdir(input_dir_path):
        print(f"ERROR: Input directory not found: {input_dir_path}")
        sys.exit(1)
    print(f"Input directory:  {input_dir_path}")

    # 2. Create Output Directory
    try:
        os.makedirs(output_dir_path, exist_ok=True)
        print(f"Output directory: {output_dir_path}")
        # Check if writable (basic check)
        test_file = os.path.join(output_dir_path, ".write_test")
        with open(test_file, "w") as f: f.write("test")
        os.remove(test_file)
    except PermissionError:
         print(f"ERROR: Permission denied to create or write to output directory: {output_dir_path}")
         sys.exit(1)
    except Exception as e:
        print(f"ERROR: Could not create output directory {output_dir_path}: {e}")
        sys.exit(1)

    # 3. Find Files to Extract
    print("Finding .pdb.gz files...")
    files_to_extract = []
    for root, _, files in os.walk(input_dir_path):
        for filename in files:
            if filename.lower().endswith(".pdb.gz"):
                files_to_extract.append(os.path.join(root, filename))

    if not files_to_extract:
        print("No .pdb.gz files found in the input directory.")
        sys.exit(0)

    print(f"Found {len(files_to_extract)} .pdb.gz files to extract.")

    # 4. Run Extraction in Parallel
    print(f"Starting extraction using up to {MAX_WORKERS} workers...")
    extracted_count = 0
    error_count = 0
    files_processed = 0
    report_interval = max(1, len(files_to_extract) // 20) # Report progress roughly every 5%

    # Using ThreadPoolExecutor for I/O-bound tasks like decompression/writing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(extract_pdb_gz, f, output_dir_path): f for f in files_to_extract}

        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            filepath = future_to_file[future]
            try:
                success, message = future.result()
                files_processed += 1
                if success:
                    extracted_count += 1
                else:
                    error_count += 1
                    print(f"ERROR: {message}") # Print errors as they happen

                # Progress reporting
                if files_processed % report_interval == 0 or files_processed == len(files_to_extract):
                    print(f"  Processed: {files_processed}/{len(files_to_extract)} (Errors: {error_count})", end='\r')

            except Exception as exc:
                error_count += 1
                files_processed += 1
                print(f"\nException processing file {filepath}: {exc}")

    print("\nExtraction process finished.") # Newline after progress indicator

    # 5. Final Summary
    end_time = time.time()
    print("--- Extraction Summary ---")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Failed/Errors:        {error_count}")
    print(f"Total files processed:  {files_processed}")
    print(f"Total time taken:     {end_time - start_time:.2f} seconds")
    print("--------------------------")

if __name__ == "__main__":
    import concurrent.futures # Make sure it's imported for the main block too
    main()