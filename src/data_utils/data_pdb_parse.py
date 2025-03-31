import os
import re
import shutil

fasta_file_path = "../../data/sequence_databases/uniprot/uniprotkb_viridiplantae.fasta"
pdb_folder_path = "../../data/structure_databases/alphafold_pdb"
output_folder_path = "../../data/structure_databases/alphafold_sp_pdb"

# Parse FASTA file to get UNIProt IDs
print(f"Reading Uniprot IDs form {fasta_file_path}")
viridiplantae_ids_set = set()
try:
    with open(fasta_file_path, 'r') as fasta_file:
        for line in fasta_file:
            try:
                parts = line.split("|")
                if len(parts) > 1:
                    uniprot_id = parts[1]
                    viridiplantae_ids_set.add(uniprot_id)
            except Exception as e:
                print("Coulde not parse")
    if not viridiplantae_ids_set:
        print("No Uniprot IDs extracted")
    else:
        print(f"Found {len(viridiplantae_ids_set)} unique uniprot Ids")
except FileNotFoundError:
    print(f"Error: Fasta file could not be found")
except Exception as e:
    print(f"Error ocurred reading the FASTA file: {e}")
    exit()

# Iterate through PDB folder and filter
print(f"Filtering PDB file in {pdb_folder_path}...")
os.makedirs(output_folder_path, exist_ok=True)
copied_count = 0
processed_count = 0

# Regex to extract UniPort ID from Alphafold filename
uniprot_pattern = re.compile(r'AF-([A-Z0-9]+)-F\d+-model_v\d+\.pdb\.gz$', re.IGNORECASE)

if not viridiplantae_ids_set:
    print("No Ids found")
else:
    # Iterate through the pdb folder
    for filename in os.listdir(pdb_folder_path):
        if filename.lower().endswith('.pdb.gz'):
            processed_count += 1
            match = uniprot_pattern.match(filename)
            if match:
                pdb_uniprot_id = match.group(1)
                # Checking id with uniprot ID
                if pdb_uniprot_id in viridiplantae_ids_set:
                    source_path = os.path.join(pdb_folder_path, filename)
                    destination_path = os.path.join(output_folder_path, filename)

                    try:
                        shutil.copy(source_path, destination_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Warning: Could not copy {filename}: {e}")
        
        if processed_count % 5000 == 0:
            print(f" Processed {processed_count} files...")

print(f"\nCompleted")
print(f"Processed {processed_count} potential PDB.gz files found in the source directory.")
print(f"Copied {copied_count} matching PDB.gz files to {output_folder_path}")

