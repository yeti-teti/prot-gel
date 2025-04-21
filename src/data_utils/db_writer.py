import os
import subprocess
import tempfile
import re
import json
import sys
import time # For progress and summary
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed # For parallelism

from Bio.PDB import PDBParser, DSSP
from Bio.SeqUtils import ProtParam
import aaindex # Keep for residue features

import numpy as np
import pandas as pd
from scipy.spatial import KDTree


# Configuration
# Data Input Paths
BASE_DATA_DIR = "../../data" 
UNIPROT_FILE = "uniprotkb_viridiplantae.txt"
PFAM_HMM_FILE = "Pfam-A.hmm"
HMMSCAN_OUT_FILE = "pfam_results.tbl"
COMPLETE_PDB_DIR = "complete_pdb_test" 
INTEGRATED_JSON_FILE = "integrated_data.json" # FINAL OUTPUT

UNIPROT_PATH = os.path.join(BASE_DATA_DIR, "sequence_databases", "uniprot", UNIPROT_FILE)
HMM_PATH = os.path.join(BASE_DATA_DIR, "sequence_databases", "pfam", PFAM_HMM_FILE)
HMMSCAN_OUTPUT = os.path.join(BASE_DATA_DIR, "sequence_databases", "pfam", HMMSCAN_OUT_FILE)
COMPLETE_PDB_FOLDER_PATH = os.path.join(BASE_DATA_DIR, "structure_databases", COMPLETE_PDB_DIR)
OUTPUT_PATH = os.path.join(BASE_DATA_DIR, INTEGRATED_JSON_FILE)


# External tool paths
DSSP_EXECUTABLE = "/opt/homebrew/bin/mkdssp" # Use the specific path provided
HMMSCAN_EXECUTABLE = "hmmscan" # Pfam scan

# Parallelism
MAX_WORKERS = os.cpu_count() 
PROGRESS_INTERVAL = 100 # Report progress every N files

# Constants
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]
BETA_SHEET_THRESHOLD = 30.0
GRAVY_THRESHOLD = 0.0
INSTABILITY_THRESHOLD = 40.0
CONTACT_THRESHOLD = 8.0



# AAIndex Setup
try:
    aa_idx = aaindex.AAIndex1()
    hydrophobicity = aa_idx['KYTJ820101']
    polarity = aa_idx['GRAR740102']
    volume = aa_idx['DAWD720101']
except Exception as e:
    print(f"FATAL: Error initializing AAIndex: {e}")
    sys.exit(1)


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
        # Ensure sequence is a string, ProtParam can fail on non-string types
        analyzer = ProtParam.ProteinAnalysis(str(sequence))
        props = {
            "molecular_weight": analyzer.molecular_weight(),
            "aromaticity": analyzer.aromaticity(),
            "instability_index": analyzer.instability_index(),
            "isoelectric_point": analyzer.isoelectric_point(),
            "gravy": analyzer.gravy(),
            "charge_at_pH_7": analyzer.charge_at_pH(7.0)
        }
        # Ensure all values are finite and serializable
        return {k: (v if np.isfinite(v) else None) for k, v in props.items()}
    except Exception:
        # Return default props with None for non-finite values if calculation fails
        return {k: (v if np.isfinite(v) else None) for k, v in default_props.items()}


def compute_residue_features(sequence):
    if not sequence: return []
    return [{'hydrophobicity': hydrophobicity.get(aa, 0.0),
             'polarity': polarity.get(aa, 0.0),
             'volume': volume.get(aa, 0.0)} for aa in sequence]


def parse_swissprot_file(file_path):
    print(f"Parsing UniProt data from: {file_path}...")
    swissprot_data = {}
    if not os.path.exists(file_path):
        print(f"ERROR: UniProt file not found: {file_path}")
        return {}
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
                                "sequence": sequence, 
                                "sequence_length": len(sequence),
                                "organism": record.get("OS", "Unknown"),
                                "taxonomy_id": tax_id.group(1) if tax_id else None,
                                "physicochemical_properties": compute_physicochemical_properties(sequence),
                                "residue_features": compute_residue_features(sequence),
                                "structural_features": [], # To be populated later
                                "domains": [] # To be populated later
                             }
                            count += 1
                            if count % 5000 == 0: print(f"  Parsed {count} UniProt records...")
                    record_lines = []
                else:
                    record_lines.append(line)
    except Exception as e:
        print(f"ERROR: Failed during UniProt parsing: {e}")
        return {} # Return empty dict on error
    print(f"Finished parsing UniProt. Total records: {len(swissprot_data)}")
    return swissprot_data

def run_hmmscan_and_parse(fasta_path, hmm_db_path, output_table_path):
    if not os.path.exists(fasta_path): # Check if input fasta exists
        print(f"ERROR: FASTA file for hmmscan not found: {fasta_path}")
        return {}
    if not os.path.exists(hmm_db_path): # Check if HMM DB exists
        print(f"ERROR: Pfam HMM DB not found: {hmm_db_path}")
        return {}

    if not os.path.exists(output_table_path):
        print(f"Running hmmscan (output to {output_table_path})...")
        cmd = [HMMSCAN_EXECUTABLE, "--domtblout", output_table_path, hmm_db_path, fasta_path]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=7200) # 2 hours timeout
            print("hmmscan completed successfully.")
        except FileNotFoundError:
            print(f"ERROR: hmmscan executable '{HMMSCAN_EXECUTABLE}' not found."); return {}
        except subprocess.CalledProcessError as e:
            print(f"ERROR: hmmscan failed:\n{e.stderr}"); return {}
        except subprocess.TimeoutExpired:
            print("ERROR: hmmscan timed out after 2 hours."); return {}
        except Exception as e:
            print(f"ERROR: Unexpected error running hmmscan: {e}"); return {}
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
                    query_name = fields[3]
                    target_name = fields[0] # Pfam accession (e.g., PFxxxxx.yy)
                    # Assuming query name format like 'sp|P12345|GENE' or just 'P12345'
                    parts = query_name.split('|')
                    uniprot_id = parts[1] if len(parts) >= 3 and parts[0] in ('sp', 'tr') else query_name

                    if not uniprot_id: continue

                    domain_info = {
                        "accession": target_name.split(".")[0], # Just PFxxxxx
                        "target_name": target_name,
                        "description": " ".join(fields[22:]),
                        "start": int(fields[17]), # Sequence start
                        "end": int(fields[18]),   # Sequence end
                        "evalue": float(fields[6]),
                        "score": float(fields[7]),
                        "bias": float(fields[8]),
                        "hmm_start": int(fields[15]),
                        "hmm_end": int(fields[16]),
                        "envelope_start": int(fields[19]),
                        "envelope_end": int(fields[20])
                    }
                    if uniprot_id not in pfam_data:
                        pfam_data[uniprot_id] = []
                    pfam_data[uniprot_id].append(domain_info)
                except (ValueError, IndexError) as parse_err:
                    continue
    except FileNotFoundError:
        print(f"ERROR: hmmscan output file not found: {hmmscan_path}"); return {}
    except Exception as e:
        print(f"ERROR: Failed parsing hmmscan output: {e}"); return {}
    print(f"Finished parsing hmmscan. Found domains for {len(pfam_data)} proteins.")
    return pfam_data

def extract_dbrefs(pdb_file_path):
    """ Parses DBREF records from a PDB file to find UniProt accessions. """
    dbrefs, uniprot_ids = [], set()
    open_func = open
    mode = 'r'
    try:
        with open_func(pdb_file_path, mode, errors='ignore') as f:
            for line in f:
                if line.startswith("DBREF") and len(line) > 41:
                     if line[26:32].strip() == "UNP":
                         try:
                             acc = line[33:41].strip()
                             if acc:
                                 dbrefs.append({
                                     "chain": line[12].strip(),
                                     "accession": acc,
                                     "db_id_code": line[42:54].strip(), # PDB ID in DBREF
                                     "pdb_start_res": int(line[14:18].strip()),
                                     "pdb_end_res": int(line[20:24].strip()),
                                     "db_start_res": int(line[55:60].strip()), # UniProt start
                                     "db_end_res": int(line[62:67].strip())  # UniProt end
                                 })
                                 uniprot_ids.add(acc)
                         except (ValueError, IndexError):
                             continue # Ignore malformed lines
    except Exception as e:
        pass
    return dbrefs, list(uniprot_ids)


def process_pdb_file(pdb_filepath, dssp_exec):
    """
    Parses a .pdb file, runs DSSP, calculates features, and links to UniProt IDs via DBREF.
    Returns list compatible with original integrate_data: [(uniprot_id, feature_dict), ...].
    """
    base_filename = os.path.basename(pdb_filepath)
    pdb_id_from_filename, ext = os.path.splitext(base_filename)
    if ext.lower() != ".pdb":
        print(f"Warning: Skipping non-.pdb file passed to process_pdb_file: {base_filename}")
        return []

    # Use extract_dbrefs to find associated UniProt IDs *first*
    dbref_details, uniprot_ids_in_file = extract_dbrefs(pdb_filepath)
    if not uniprot_ids_in_file:
        return [] # Return empty list, like original parse_single_pdb

    results_for_integration = []
    structural_feature = {
        "pdb_id": pdb_id_from_filename, # ID from filename
        "pdb_file": base_filename,
        "dbref_records": dbref_details, # Store the parsed DBREF info
        "helix_percentage": None,
        "sheet_percentage": None,
        "coil_percentage": None,
        "total_residues_dssp": 0,
        "dssp_residue_details": [],
        "ca_coordinates": [],
        "contact_map_indices_ca": [],
        "processing_error": None # Add field for errors during processing
    }

    try:
        # Parse PDB Structure
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id_from_filename, pdb_filepath)
        if not structure or len(structure) == 0:
             structural_feature["processing_error"] = "Biopython parse error or empty structure" 
             return [(uid, structural_feature) for uid in uniprot_ids_in_file]
        model = structure[0]

        # Run DSSP
        dssp_data = []
        try:
            dssp = DSSP(model, pdb_filepath, dssp=dssp_exec)
            dssp_data = list(dssp)
        except FileNotFoundError:
             structural_feature["processing_error"] = f"DSSP executable not found at: {dssp_exec}"
             return [(uid, structural_feature) for uid in uniprot_ids_in_file]
        except Exception as dssp_err:
            error_detail = str(dssp_err)
            if isinstance(dssp_err, subprocess.CalledProcessError) and dssp_err.stderr:
                 error_detail += f" | DSSP stderr: {dssp_err.stderr.decode(errors='ignore')[:200]}"
            structural_feature["processing_error"] = f"DSSP execution failed: {error_detail}"

        # Process DSSP Results
        residue_details = []
        helix_count, sheet_count, total_residues = 0, 0, 0
        if dssp_data:
            total_residues = len(dssp_data)
            for dssp_key, dssp_info in dssp_data:
                res_id = dssp_key[1]
                ss = dssp_info[1]
                rel_acc = dssp_info[2] if np.isfinite(dssp_info[2]) else None
                phi = dssp_info[3] if np.isfinite(dssp_info[3]) else None
                psi = dssp_info[4] if np.isfinite(dssp_info[4]) else None
                residue_details.append({
                    "chain": dssp_key[0], "residue_seq": res_id[1], "residue_icode": res_id[2].strip(),
                    "amino_acid": dssp_info[0], "secondary_structure": ss,
                    "relative_accessibility": rel_acc, "phi": phi, "psi": psi
                })
                if ss in ('H', 'G', 'I'): helix_count += 1
                elif ss in ('E', 'B'): sheet_count += 1

            structural_feature["helix_percentage"] = (helix_count / total_residues * 100) if total_residues else 0
            structural_feature["sheet_percentage"] = (sheet_count / total_residues * 100) if total_residues else 0
            structural_feature["coil_percentage"] = ((total_residues - helix_count - sheet_count) / total_residues * 100) if total_residues else 0
            structural_feature["total_residues_dssp"] = total_residues
            structural_feature["dssp_residue_details"] = residue_details
        elif not structural_feature["processing_error"]:
             structural_feature["processing_error"] = "DSSP returned no data"

        # Calculate CA Coordinates and Contact Map
        ca_coordinates = []
        contact_map_indices = []
        try:
            ca_atoms = [a for a in model.get_atoms() if a.get_name() == 'CA']
            if len(ca_atoms) > 1:
                ca_coords_np = np.array([a.get_coord() for a in ca_atoms])
                ca_coordinates = [{"index": i, "chain": a.get_parent().get_parent().id,
                                   "residue_seq": a.get_parent().id[1], "residue_icode": a.get_parent().id[2].strip(),
                                   "x": float(c[0]), "y": float(c[1]), "z": float(c[2])}
                                  for i, (a, c) in enumerate(zip(ca_atoms, ca_coords_np))]
                tree = KDTree(ca_coords_np)
                pairs = list(tree.query_pairs(r=CONTACT_THRESHOLD))
                contact_map_indices = [{"idx": i1, "idx2": i2} for i1, i2 in pairs] 

            structural_feature["ca_coordinates"] = ca_coordinates
            structural_feature["contact_map_indices_ca"] = contact_map_indices
        except Exception as geom_err:
             error_msg = f"Geometry calculation failed: {geom_err}"
             structural_feature["contact_map_indices_ca"] = [] # Ensure empty list on error
             structural_feature["ca_coordinates"] = []
             if structural_feature["processing_error"]:
                 structural_feature["processing_error"] += f"; {error_msg}"
             else:
                 structural_feature["processing_error"] = error_msg

        # Create the list of tuples to return, linking feature dict to each UniProt ID
        results_for_integration = [(uid, structural_feature) for uid in uniprot_ids_in_file]
        return results_for_integration

    except Exception as general_err:
        err_msg = f"Unexpected error in process_pdb_file: {general_err}"
        structural_feature["processing_error"] = err_msg
        return [(uid, structural_feature) for uid in uniprot_ids_in_file]


def integrate_data(uniprot_data, pfam_data, structure_data_map):
    print("Integrating data sources...")
    integrated_count = 0
    gelation_count = 0
    no_structure_count = 0

    # Make sure uniprot_data is a dictionary
    if not isinstance(uniprot_data, dict):
        print("ERROR: uniprot_data is not a dictionary in integrate_data.")
        return {}

    for uniprot_id, protein_info in uniprot_data.items():
        protein_info["domains"] = pfam_data.get(uniprot_id, [])
        # structure_data_map maps uniprot_id -> list of structural_feature dicts
        protein_info["structural_features"] = structure_data_map.get(uniprot_id, [])
        integrated_count += 1

        if not protein_info["structural_features"]:
            no_structure_count +=1

        # Gelation prediction logic 
        domains = protein_info["domains"]
        structs = protein_info["structural_features"]
        props = protein_info.get("physicochemical_properties", {})

        gel_domain = any(d.get("accession") in GELATION_DOMAINS for d in domains)
        # Apply threshold only if sheet_percentage is not None
        gel_struct = any(s.get("sheet_percentage") is not None and s["sheet_percentage"] > BETA_SHEET_THRESHOLD for s in structs)

        # Check if props exist and keys are present before accessing
        gel_seq = False
        if props:
             gravy_val = props.get("gravy")
             instability_val = props.get("instability_index")
             if gravy_val is not None and instability_val is not None:
                  gel_seq = (gravy_val > GRAVY_THRESHOLD and instability_val > INSTABILITY_THRESHOLD)

        protein_info["gelation"] = "yes" if (gel_domain or gel_struct or gel_seq) else "no"
        if protein_info["gelation"] == "yes":
            gelation_count += 1

    print(f"Integration complete. Processed: {integrated_count} UniProt entries.")
    print(f"  Entries w/o structural features: {no_structure_count}")
    print(f"  Predicted gelating: {gelation_count}")
    return uniprot_data


# Main Execution
def main():
    print("--- Starting Streamlined Protein Data Integration (JSON Output)")
    overall_start_time = time.time()

    # Step 1: Parse UniProt Data (.txt file format assumed)
    uniprot_data = parse_swissprot_file(UNIPROT_PATH)
    if not uniprot_data:
        print("ERROR: Failed to parse UniProt data. Exiting.")
        sys.exit(1)

    # Step 2: Parse PFAM domains (Run hmmscan if needed)
    pfam_data = {}
    # Check if HMM path exists before trying to run hmmscan
    if os.path.exists(HMM_PATH):
         # Need a FASTA file derived from uniprot_data for hmmscan
         # Let's create a temporary fasta file if run_hmmscan_and_parse needs it
         # Or modify run_hmmscan_and_parse if it can take sequence data directly (unlikely for external tool)

         # Create temp FASTA for hmmscan
         temp_fasta_path = None
         try:
              with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".fasta") as temp_fasta:
                   temp_fasta_path = temp_fasta.name
                   fasta_count = 0
                   for uid, data in uniprot_data.items():
                        if "sequence" in data and data["sequence"]:
                             temp_fasta.write(f">sp|{uid}|Generated\n{data['sequence']}\n")
                             fasta_count += 1
                   print(f"Generated temporary FASTA file for hmmscan with {fasta_count} sequences: {temp_fasta_path}")

              if fasta_count > 0:
                    pfam_data = run_hmmscan_and_parse(temp_fasta_path, HMM_PATH, HMMSCAN_OUTPUT)
              else:
                  print("Warning: No sequences found in UniProt data to run hmmscan.")

         finally:
              if temp_fasta_path and os.path.exists(temp_fasta_path):
                  try: os.remove(temp_fasta_path)
                  except OSError: print(f"Warning: Could not remove temp FASTA file {temp_fasta_path}")
         # End temp FASTA

    else:
        print(f"Warning: Pfam HMM DB not found: {HMM_PATH}. Skipping domain parsing.")


    # Step 3: Process PDB files from 'complete_pdb' directory (Parallel DSSP etc.)
    print(f"\n--- Processing PDB Structures from '{COMPLETE_PDB_DIR}'")
    pdb_process_start_time = time.time()
    combined_pdb_data = {} # Maps uniprot_id -> list of structural_feature dicts
    processed_pdb_count = 0
    error_in_pdb_processing = 0
    pdb_files_to_process = []

    if not os.path.isdir(COMPLETE_PDB_FOLDER_PATH):
        print(f"WARNING: Input PDB directory not found: {COMPLETE_PDB_FOLDER_PATH}. Skipping structure processing.")
    else:
        print(f"Finding .pdb files in {COMPLETE_PDB_FOLDER_PATH}...")
        for root, _, files in os.walk(COMPLETE_PDB_FOLDER_PATH):
            for filename in files:
                if filename.lower().endswith(".pdb"):
                    pdb_files_to_process.append(os.path.join(root, filename))

        if not pdb_files_to_process:
            print("No .pdb files found to process.")
        else:
            print(f"Found {len(pdb_files_to_process)} .pdb files. Starting parallel processing...")
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_filepath = {executor.submit(process_pdb_file, f, DSSP_EXECUTABLE): f for f in pdb_files_to_process}

                for future in as_completed(future_to_filepath):
                    filepath = future_to_filepath[future]
                    try:
                        # process_pdb_file returns list: [(uid, feature), ...] or []
                        results_list = future.result()
                        processed_pdb_count += 1

                        if results_list: # If list is not empty
                            file_had_error = False
                            for uniprot_id, feature_dict in results_list:
                                if feature_dict.get("processing_error"):
                                     file_had_error = True
                                # Append the single feature dict to the list for that UniProt ID
                                combined_pdb_data.setdefault(uniprot_id, []).append(feature_dict)
                            if file_had_error:
                                error_in_pdb_processing += 1

                        # Progress Reporting
                        if processed_pdb_count % PROGRESS_INTERVAL == 0 or processed_pdb_count == len(pdb_files_to_process):
                            print(f"  Processed PDBs: {processed_pdb_count}/{len(pdb_files_to_process)} (Errors*: {error_in_pdb_processing})", end='\r')

                    except Exception as exc:
                        # Catch critical errors in the future itself
                        processed_pdb_count += 1
                        error_in_pdb_processing += 1
                        print(f"\nCritical error processing file task {os.path.basename(filepath)}: {exc}")
                        # Optionally log this critical failure associated with the file path

            print("\nPDB processing finished.") # Newline after progress

    pdb_process_end_time = time.time()
    print(f"PDB Processing Summary: Processed={processed_pdb_count}, Files w/ Errors={error_in_pdb_processing}, Found structures for={len(combined_pdb_data)} UniProt IDs")
    print(f"PDB processing time: {pdb_process_end_time - pdb_process_start_time:.2f} seconds")


    # Step 4: Integrate All Data
    integrated_data = integrate_data(uniprot_data, pfam_data, combined_pdb_data)


    # Step 5: Save Final Integrated Data to JSON
    print(f"\n--- Saving Integrated Data to JSON")
    if not integrated_data:
         print("ERROR: No integrated data produced. Skipping JSON save.")
    else:
         print(f"Saving integrated data ({len(integrated_data)} entries) to {OUTPUT_PATH}...")
         try:
             # Ensure parent directory exists
             os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
             with open(OUTPUT_PATH, "w") as f:
                  json.dump(integrated_data, f, indent=2)
             print("Integrated data saved successfully.")
         except TypeError as e:
              print(f"ERROR: Could not write integrated data JSON due to non-serializable data: {e}")
              print("Consider using a custom JSON encoder if numpy types are present.")
         except Exception as e:
              print(f"ERROR: Could not write integrated data JSON: {e}")

    overall_end_time = time.time()
    print("\n--- Pipeline Finished")
    print(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds")


if __name__ == "__main__":
    main()