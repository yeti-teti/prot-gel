import os
import subprocess
import re
import json
import tempfile
import shutil
import gzip
import sys
from collections import Counter

from Bio.PDB import PDBParser, DSSP
from Bio.SeqUtils import ProtParam
import aaindex

import numpy as np
import pandas as pd 
from scipy.spatial import KDTree

import pyarrow as pa 
import pyarrow.parquet as pq 
import pyarrow.fs as pafs 

import dask
from dask import delayed, compute
from dotenv import load_dotenv 


# --- Configuration ---
# Data Input Paths
BASE_DATA_DIR = "../../data" # Adjust base path as needed
UNIPROT_FILE = "uniprotkb_viridiplantae.txt"
PFAM_HMM_FILE = "Pfam-A.hmm"
HMMSCAN_OUT_FILE = "pfam_results.tbl"
PDB_DIR = "pdb_data"
ALPHAFOLD_DIR = "alphafold_sp_pdb"
INTEGRATED_JSON_PATH = "integrated_data.json"

UNIPROT_FASTA_PATH = os.path.join(BASE_DATA_DIR, "sequence_databases", "uniprot", UNIPROT_FILE)
HMM_PATH = os.path.join(BASE_DATA_DIR, "sequence_databases", "pfam", PFAM_HMM_FILE)
HMMSCAN_OUTPUT = os.path.join(BASE_DATA_DIR, "sequence_databases", "pfam", HMMSCAN_OUT_FILE)
PDB_FOLDER_PATH = os.path.join(BASE_DATA_DIR, "structure_databases", PDB_DIR)
ALPHAFOLD_PATH = os.path.join(BASE_DATA_DIR, "structure_databases", ALPHAFOLD_DIR)

# External tool paths
DSSP_EXECUTABLE = "dssp"
HMMSCAN_EXECUTABLE = "hmmscan"

# Cloudflare R2 Output Configuration
ENV_FILE_PATH = ".env" # Assumes .env file in the script's directory
# R2 Target path
R2_OUTPUT_DIR = "integrated_data/viridiplantae_dataset_partitioned"

# Constants
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]
BETA_SHEET_THRESHOLD = 30.0
GRAVY_THRESHOLD = 0.0
INSTABILITY_THRESHOLD = 40.0
CONTACT_THRESHOLD = 8.0

# --- Load Environment Variables for R2 ---
print(f"Loading R2 credentials from: {ENV_FILE_PATH}")
if not load_dotenv(dotenv_path=ENV_FILE_PATH):
    print(f"Warning: .env file not found at {ENV_FILE_PATH}. Using system environment variables.")

r2_access_key = os.getenv("CLOUDFARE_ACCESS_KEY")
r2_secret_key = os.getenv("CLOUDFARE_SECRET_KEY")
r2_account_id = os.getenv("CLOUDFARE_ACCOUNT_ID")
r2_bucket_name = os.getenv("CLOUDFARE_BUCKET_NAME")
r2_endpoint = os.getenv("CLOUDFARE_ENDPOINT")

if not r2_endpoint and r2_account_id:
    r2_endpoint = f"https://{r2_account_id}.r2.cloudflarestorage.com"

# Validate required R2 variables
if not all([r2_access_key, r2_secret_key, r2_bucket_name, r2_endpoint]):
    print("ERROR: Missing Cloudflare R2 credentials/config (KEY, SECRET, BUCKET, ENDPOINT/ACCOUNT_ID) in environment/.env")
    sys.exit(1)
print(f"Target R2 Endpoint: {r2_endpoint}, Bucket: {r2_bucket_name}")


# --- AAIndex Setup ---
try:
    aa_idx = aaindex.AAIndex1()
    hydrophobicity = aa_idx['KYTJ820101']
    polarity = aa_idx['GRAR740102']
    volume = aa_idx['DAWD720101']
except Exception as e:
    print(f"FATAL: Error initializing AAIndex: {e}")
    sys.exit(1)

# --- Helper Functions ---
def parse_record(record_lines):
    record_data, current_tag, in_sequence, sequence_lines = {}, None, False, []
    for line in record_lines:
        if not in_sequence and len(line) >= 5 and line[0:2].isalpha() and line[2:5] == "   ":
            tag, content = line[0:2], line[5:].rstrip()
            if tag == "SQ": 
                in_sequence = True
            else: 
                current_tag = tag
                record_data[tag] = record_data.get(tag, "") + (" " if tag in record_data else "") + content
        elif in_sequence:
            if line.startswith("//"): 
                in_sequence = False
            else: 
                sequence_lines.append(re.sub(r"[\s\d]", "", line))
        elif current_tag and line.startswith("     ") and current_tag in record_data:
            record_data[current_tag] += " " + line.strip()
    if sequence_lines: 
        record_data["SQ"] = "".join(sequence_lines)
    return record_data

def compute_physicochemical_properties(sequence):
    default_props = {"molecular_weight": 0.0, "aromaticity": 0.0, "instability_index": 0.0,
                     "isoelectric_point": 0.0, "gravy": 0.0, "charge_at_pH_7": 0.0}
    if not isinstance(sequence, str) or not sequence: return default_props
    try:
        analyzer = ProtParam.ProteinAnalysis(str(sequence))
        return {"molecular_weight": analyzer.molecular_weight(), "aromaticity": analyzer.aromaticity(),
                "instability_index": analyzer.instability_index(), "isoelectric_point": analyzer.isoelectric_point(),
                "gravy": analyzer.gravy(), "charge_at_pH_7": analyzer.charge_at_pH(7.0)}
    except Exception: return default_props

def compute_aa_composition(sequence):
    if not sequence: return {}
    count = Counter(sequence); total = len(sequence)
    return {aa: c / total for aa, c in count.items()}

def compute_residue_features(sequence):
    return [{'hydrophobicity': hydrophobicity.get(aa, 0.0),
             'polarity': polarity.get(aa, 0.0),
             'volume': volume.get(aa, 0.0)} for aa in sequence]

def parse_swissprot_file(file_path):
    print(f"Parsing UniProt data from: {file_path}...")
    swissprot_data = {}
    try:
        with open(file_path, "r") as f:
            record_lines = []; count = 0
            for line in f:
                line = line.rstrip("\n")
                if line == "//":
                    if record_lines:
                        record = parse_record(record_lines)
                        uniprot_id = record.get("AC", "").split(";")[0].strip()
                        sequence = record.get("SQ", "")
                        if uniprot_id and sequence:
                            tax_id = re.search(r"NCBI_TaxID=(\d+)", record.get("OX", ""))
                            swissprot_data[uniprot_id] = {
                                "sequence": sequence, "sequence_length": len(sequence),
                                "organism": record.get("OS", "Unknown"), "taxonomy_id": tax_id.group(1) if tax_id else None,
                                "physicochemical_properties": compute_physicochemical_properties(sequence),
                                "aa_composition": compute_aa_composition(sequence),
                                "residue_features": compute_residue_features(sequence),
                                "structural_features": [], "domains": [] }
                            count += 1
                            if count % 5000 == 0: print(f"  Parsed {count} UniProt records...")
                    record_lines = []
                else: record_lines.append(line)
    except FileNotFoundError: print(f"ERROR: UniProt file not found: {file_path}"); return {}
    except Exception as e: print(f"ERROR: Failed during UniProt parsing: {e}"); return {}
    print(f"Finished parsing UniProt. Total records: {len(swissprot_data)}")
    return swissprot_data

def run_hmmscan_and_parse(fasta_path, hmm_db_path, output_table_path):
    if not os.path.exists(output_table_path):
        print(f"Running hmmscan (output to {output_table_path})...")
        cmd = [HMMSCAN_EXECUTABLE, "--domtblout", output_table_path, hmm_db_path, fasta_path]
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        except FileNotFoundError: print(f"ERROR: hmmscan executable '{HMMSCAN_EXECUTABLE}' not found."); return {}
        except subprocess.CalledProcessError as e: print(f"ERROR: hmmscan failed:\n{e.stderr}"); return {}
        except subprocess.TimeoutExpired: print("ERROR: hmmscan timed out."); return {}
        except Exception as e: print(f"ERROR: Unexpected error running hmmscan: {e}"); return {}
    else:
        print(f"Using existing hmmscan output: {output_table_path}")
    return parse_hmmscan_table(output_table_path)

def parse_hmmscan_table(hmmscan_path):
    print(f"Parsing hmmscan output from {hmmscan_path}...")
    pfam_data = {}
    try:
        with open(hmmscan_path, "r") as f:
            for line in f:
                if line.startswith("#"): continue
                fields = line.split()
                if len(fields) < 22: continue
                try:
                    query_name = fields[3]; target_name = fields[0]
                    parts = query_name.split('|'); uniprot_id = parts[1] if len(parts) >= 2 else query_name
                    if not uniprot_id: continue
                    domain_info = {
                        "accession": target_name.split(".")[0], "target_name": target_name,
                        "description": " ".join(fields[22:]), "start": int(fields[17]), "end": int(fields[18]),
                        "evalue": float(fields[6]), "score": float(fields[7]), "bias": float(fields[8]),
                        "hmm_start": int(fields[15]), "hmm_end": int(fields[16]),
                        "envelope_start": int(fields[19]), "envelope_end": int(fields[20])}
                    if uniprot_id not in pfam_data: pfam_data[uniprot_id] = []
                    pfam_data[uniprot_id].append(domain_info)
                except (ValueError, IndexError): continue
    except FileNotFoundError: print(f"ERROR: hmmscan output file not found: {hmmscan_path}"); return {}
    except Exception as e: print(f"ERROR: Failed parsing hmmscan output: {e}"); return {}
    print(f"Finished parsing hmmscan. Found domains for {len(pfam_data)} proteins.")
    return pfam_data

def extract_dbrefs(pdb_file_path):
    dbrefs, uniprot_ids = [], set()
    open_func = gzip.open if pdb_file_path.lower().endswith('.gz') else open
    mode = 'rt' if pdb_file_path.lower().endswith('.gz') else 'r'
    try:
        with open_func(pdb_file_path, mode) as f:
            for line in f:
                if line.startswith("DBREF") and line[26:32].strip() == "UNP":
                    try:
                        acc = line[33:41].strip()
                        if acc:
                            dbrefs.append({ "chain": line[12].strip(), "accession": acc,
                                            "db_id_code": line[42:54].strip(), "pdb_start_res": int(line[14:18].strip()),
                                            "pdb_end_res": int(line[20:24].strip()), "db_start_res": int(line[55:60].strip()),
                                            "db_end_res": int(line[62:67].strip()) })
                            uniprot_ids.add(acc)
                    except (ValueError, IndexError): continue
    except Exception: pass
    return dbrefs, list(uniprot_ids)

def parse_single_pdb(pdb_path):
    pdb_id = os.path.basename(pdb_path).split(".")[0]
    temp_pdb_for_dssp = None
    try:
        if pdb_path.endswith('.gz'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='wt') as tmp_file:
                with gzip.open(pdb_path, 'rt') as f_in: shutil.copyfileobj(f_in, tmp_file)
                temp_pdb_for_dssp = tmp_file.name
            processing_path = temp_pdb_for_dssp
        else: processing_path = pdb_path

        dbref_details, uniprot_ids = extract_dbrefs(pdb_path)
        if not uniprot_ids: return []

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, processing_path)
        model = structure[0]

        disulfide_bonds = []
        open_func = gzip.open if pdb_path.endswith('.gz') else open
        mode = 'rt' if pdb_path.endswith('.gz') else 'r'
        with open_func(pdb_path, mode) as f:
            for line in f:
                 if line.startswith("SSBOND"):
                     try: disulfide_bonds.append({"chain1": line[15].strip(), "res1": int(line[17:21].strip()),
                                                   "chain2": line[29].strip(), "res2": int(line[31:35].strip())})
                     except (ValueError, IndexError): continue

        residue_details, helix_count, sheet_count, total_residues = [], 0, 0, 0
        try:
            dssp = DSSP(model, processing_path, dssp=DSSP_EXECUTABLE)
            dssp_data = list(dssp); total_residues = len(dssp_data)
            for dssp_key, dssp_info in dssp_data:
                res_id = dssp_key[1]; ss = dssp_info[1]
                residue_details.append({ "chain": dssp_key[0], "residue_seq": res_id[1], "residue_icode": res_id[2].strip(),
                                         "amino_acid": dssp_info[0], "secondary_structure": ss,
                                         "relative_accessibility": dssp_info[2], "phi": dssp_info[3], "psi": dssp_info[4]})
                if ss in ('H', 'G', 'I'): helix_count += 1
                elif ss in ('E', 'B'): sheet_count += 1
        except Exception: pass

        contact_map, ca_coordinates = [], []
        try:
            ca_atoms = [a for a in model.get_atoms() if a.get_name() == 'CA']
            if len(ca_atoms) > 1:
                ca_coords = np.array([a.get_coord() for a in ca_atoms])
                residue_ids = [(a.get_parent().get_parent().id, a.get_parent().get_id()) for a in ca_atoms]
                ca_coordinates = [{"chain": r_id[0], "residue_seq": r_id[1][1], "residue_icode": r_id[1][2].strip(),
                                  "x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                                  for r_id, c in zip(residue_ids, ca_coords)]
                tree = KDTree(ca_coords)
                pairs = tree.query_pairs(r=CONTACT_THRESHOLD)
                contact_map = [(residue_ids[i], residue_ids[j]) for i, j in pairs]
        except Exception: pass

        helix_p = (helix_count / total_residues * 100) if total_residues else 0
        sheet_p = (sheet_count / total_residues * 100) if total_residues else 0
        coil_p = ((total_residues - helix_count - sheet_count) / total_residues * 100) if total_residues else 0
        structural_feature = {
            "pdb_id": pdb_id, "pdb_file": os.path.basename(pdb_path), "dbref_records": dbref_details,
            "helix_percentage": helix_p, "sheet_percentage": sheet_p, "coil_percentage": coil_p,
            "residue_details": residue_details, "disulfide_bonds": disulfide_bonds,
            "contact_map_ca": contact_map, "ca_coordinates": ca_coordinates}

        return [(uid, structural_feature) for uid in uniprot_ids]
    except Exception: return []
    finally:
        if temp_pdb_for_dssp and os.path.exists(temp_pdb_for_dssp):
            try: os.remove(temp_pdb_for_dssp)
            except OSError: pass

def find_pdb_files(root_dir):
    pdb_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".pdb", ".pdb.gz")):
                pdb_files.append(os.path.join(subdir, file))
    return pdb_files

def integrate_data(uniprot_data, pfam_data, pdb_data_map):
    print("Integrating data sources...")
    gelation_count = 0
    for uniprot_id, protein_info in uniprot_data.items():
        protein_info["domains"] = pfam_data.get(uniprot_id, [])
        protein_info["structural_features"] = pdb_data_map.get(uniprot_id, [])

        domains = protein_info["domains"]; structs = protein_info["structural_features"]
        props = protein_info.get("physicochemical_properties", {})
        gel_domain = any(d.get("accession") in GELATION_DOMAINS for d in domains)
        gel_struct = any(f.get("sheet_percentage", 0) > BETA_SHEET_THRESHOLD for f in structs)
        gel_seq = (props.get("gravy", -999) > GRAVY_THRESHOLD and
                   props.get("instability_index", -999) > INSTABILITY_THRESHOLD)
        protein_info["gelation"] = "yes" if (gel_domain or gel_struct or gel_seq) else "no"
        if protein_info["gelation"] == "yes": gelation_count += 1
    print(f"Integration complete. Predicted gelating: {gelation_count} / {len(uniprot_data)}")
    return uniprot_data

# --- Main Execution ---
def main():
    print("--- Starting Protein Data Integration & Cloud Upload Pipeline ---")

    # Step 1: Parse UniProt
    uniprot_data = parse_swissprot_file(UNIPROT_FASTA_PATH)
    if not uniprot_data: sys.exit(1)

    # Step 2: Parse PFAM domains
    pfam_data = {}
    if os.path.exists(HMM_PATH):
        pfam_data = run_hmmscan_and_parse(UNIPROT_FASTA_PATH, HMM_PATH, HMMSCAN_OUTPUT)
    else: print(f"Warning: Pfam HMM DB not found: {HMM_PATH}. Skipping domains.")

    # Step 3: Parse Experimental PDBs (Parallel using Dask)
    print("Parsing experimental PDB files...")
    exp_pdb_data = {}
    if os.path.isdir(PDB_FOLDER_PATH):
        exp_pdb_files = find_pdb_files(PDB_FOLDER_PATH)
        if exp_pdb_files:
            delayed_tasks = [delayed(parse_single_pdb)(f) for f in exp_pdb_files]
            print(f"  Submitting {len(delayed_tasks)} exp PDB tasks to Dask...")
            results = compute(*delayed_tasks)
            print("  Dask finished exp PDBs.")
            for res_list in results:
                for uid, feat in res_list:
                    if uid not in exp_pdb_data: exp_pdb_data[uid] = []
                    exp_pdb_data[uid].append(feat)
            print(f"  Parsed exp PDBs -> {len(exp_pdb_data)} UniProt IDs.")
        else: print("No experimental PDB files found.")
    else: print(f"Warning: Exp PDB folder not found: {PDB_FOLDER_PATH}")

    # Step 4: Parse AlphaFold PDBs (Parallel using Dask, for missing)
    print("Parsing AlphaFold PDB files (where needed)...")
    alphafold_pdb_data = {}
    ids_needing_structure = set(uniprot_data.keys()) - set(exp_pdb_data.keys())
    if os.path.isdir(ALPHAFOLD_PATH) and ids_needing_structure:
        af_files_to_parse = []
        for uid in ids_needing_structure:
             fpath = os.path.join(ALPHAFOLD_PATH, f"AF-{uid}-F1-model_v4.pdb.gz")
             if os.path.exists(fpath): af_files_to_parse.append(fpath)
        if af_files_to_parse:
            delayed_tasks = [delayed(parse_single_pdb)(f) for f in af_files_to_parse]
            print(f"  Submitting {len(delayed_tasks)} AF PDB tasks to Dask...")
            results = compute(*delayed_tasks)
            print("  Dask finished AF PDBs.")
            for res_list in results:
                for uid, feat in res_list:
                    if uid not in alphafold_pdb_data: alphafold_pdb_data[uid] = []
                    alphafold_pdb_data[uid].append(feat)
            print(f"  Parsed AF PDBs -> {len(alphafold_pdb_data)} UniProt IDs.")
        else: print("No AlphaFold files found for remaining proteins.")
    elif not ids_needing_structure: print("No proteins require AlphaFold structures.")
    else: print(f"Warning: AlphaFold PDB folder not found: {ALPHAFOLD_PATH}")

    # Step 5: Combine Structural Data
    combined_pdb_data = exp_pdb_data.copy()
    for uid, feats in alphafold_pdb_data.items():
        if uid not in combined_pdb_data: combined_pdb_data[uid] = feats
    print(f"Combined structures for {len(combined_pdb_data)} proteins.")

    # Step 6: Integrate All Data (Returns Python Dict)
    integrated_data = integrate_data(uniprot_data, pfam_data, combined_pdb_data)

    # Step 7: Convert integrated data and write directly to R2 Parquet
    print("--- Step 7: Converting Data and Writing to R2 Parquet ---")
    if not integrated_data:
        print("ERROR: No integrated data to write to Parquet.")
        sys.exit(1)

    try:
        # Convert dict {uniprot_id: data} to DataFrame rows
        print("Converting integrated data dictionary to Pandas DataFrame...")
        print("WARNING: This step requires significant RAM for large datasets!")
        df = pd.DataFrame.from_dict(integrated_data, orient='index')
        df = df.reset_index().rename(columns={'index': 'uniprot_id'})
        print(f"DataFrame created with shape: {df.shape}")

        # Add partitioning column
        df['uniprot_id'] = df['uniprot_id'].astype(str)
        df['uniprot_id_prefix'] = df['uniprot_id'].str[0].fillna('?')

        # Convert to Arrow Table
        print("Converting DataFrame to PyArrow Table...")
        table = pa.Table.from_pandas(df, preserve_index=False)
        print("PyArrow Table created.")

        # Configure R2 Filesystem
        print("Configuring R2 filesystem connection...")
        r2_fs = pafs.S3FileSystem(
            endpoint_override=r2_endpoint, access_key=r2_access_key, secret_key=r2_secret_key, scheme="https")

        # Write Parquet Dataset to R2
        full_dataset_uri = f"{r2_bucket_name}/{R2_OUTPUT_DIR}"
        print(f"Writing partitioned Parquet dataset to: {full_dataset_uri}")
        pq.write_to_dataset(
            table, root_path=full_dataset_uri, partition_cols=['uniprot_id_prefix'],
            filesystem=r2_fs, use_threads=True, existing_data_behavior='overwrite_or_ignore')
        print("Successfully wrote Parquet dataset to R2.")

    except MemoryError:
        print("\n\nERROR: Ran out of memory during DataFrame/Arrow Table conversion.")
        print("The dataset is too large to process in memory with this script.")
        print("Consider using a machine with more RAM or redesigning the pipeline")
        print("to process data incrementally (e.g., using Dask DataFrames throughout).\n\n")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed during data conversion or R2 Parquet writing: {e}")
        # import traceback; traceback.print_exc()
        sys.exit(1)

    # --- JSON Saving---
    print(f"Saving integrated data to {INTEGRATED_JSON_PATH}...")
    try:
        os.makedirs(os.path.dirname(INTEGRATED_JSON_PATH), exist_ok=True)
        with open(INTEGRATED_JSON_PATH, "w") as f: json.dump(integrated_data, f, indent=2)
        print("Integrated data saved successfully.")
    except Exception as e: print(f"ERROR: Could not write integrated data JSON: {e}")

    print("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()