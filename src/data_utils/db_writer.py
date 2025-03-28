import os
import subprocess
from Bio.PDB import PDBParser, DSSP
from Bio.SeqUtils import ProtParam
from collections import Counter
import re
import json
import aaindex
import numpy as np
from scipy.spatial import KDTree
import tempfile
import shutil
import gzip
import dask
from dask import delayed, compute
from mondo_db import write

# Constants for gelation analysis
GELATION_DOMAINS = ["PF00190", "PF04702", "PF00234"]
BETA_SHEET_THRESHOLD = 30.0  # Threshold for beta-sheet percentage
GRAVY_THRESHOLD = 0.0  # Threshold for hydrophobicity
INSTABILITY_THRESHOLD = 40.0  # Threshold for instability index
CONTACT_THRESHOLD = 8.0  # Distance threshold for residue contacts (Ångstroms)

# aaindex initialization to get specific indices
aa = aaindex.AAIndex1()
hydrophobicity = aa['KYTJ820101']  # Kyte & Doolittle hydrophobicity
polarity = aa['GRAR740102']        # Grantham polarity
volume = aa['DAWD720101']          # Dawid volume

# Swiss-Prot
def parse_record(record_lines):
    """
    Parses a single SwissProt record (a list of lines) into a dictionary.
    Extracts multi-line entries for tags (e.g., AC, DE, OS) and captures the sequence from the SQ section.
    """
    record_data = {}
    current_tag = None
    in_sequence = False
    sequence_lines = []

    for line in record_lines:
        if not in_sequence and len(line) >= 5 and line[0:2].isalpha() and line[2:5] == "   ":
            tag = line[0:2]
            content = line[5:].rstrip()
            if tag == "SQ":
                in_sequence = True
            else:
                current_tag = tag
                if tag in record_data:
                    record_data[tag] += " " + content
                else:
                    record_data[tag] = content
        elif in_sequence:
            if line.startswith("//"):
                in_sequence = False
            else:
                seq_line = re.sub(r"[\s\d]", "", line)
                sequence_lines.append(seq_line)
        else:
            if current_tag:
                record_data[current_tag] += " " + line.strip()

    if sequence_lines:
        record_data["SQ"] = "".join(sequence_lines)
    return record_data

def parse_swissprot_file_line_by_line(file_path):
    """
    Parses a SwissProt flat file line by line, extracting key fields and computing additional properties.
    Adds sequence_length to the data dictionary.
    """
    swissprot_data = {}
    with open(file_path, "r") as f:
        record_lines = []
        for line in f:
            line = line.rstrip("\n")
            if line == "//":
                record = parse_record(record_lines)
                uniprot_id = record.get("AC", "unknown").split(";")[0].strip()
                sequence = record.get("SQ", "")
                organism = record.get("OS", "").strip()
                taxonomy_id = None
                if "OX" in record:
                    match = re.search(r"(\d+)", record["OX"])
                    if match:
                        taxonomy_id = match.group(1)

                properties = compute_physicochemical_properties(sequence)
                aa_composition = compute_aa_composition(sequence)

                swissprot_data[uniprot_id] = {
                    "sequence": sequence,
                    "sequence_length": len(sequence),  # Added amino acid length
                    "organism": organism,
                    "taxonomy_id": taxonomy_id,
                    "physicochemical_properties": properties,
                    "aa_composition": aa_composition,
                    "residue_features": compute_residue_features(sequence),
                    "structural_features": [],
                    "domains": []
                }
                record_lines = []
            else:
                record_lines.append(line)
    return swissprot_data

def compute_physicochemical_properties(sequence):
    """
    Computes physicochemical properties of a protein sequence.
    Adds charge at pH 7.0.
    """
    try:
        analyzer = ProtParam.ProteinAnalysis(sequence)
        properties = {
            "molecular_weight": analyzer.molecular_weight(),
            "aromaticity": analyzer.aromaticity(),
            "instability_index": analyzer.instability_index(),
            "isoelectric_point": analyzer.isoelectric_point(),
            "gravy": analyzer.gravy(),
            "secondary_structure_fraction": analyzer.secondary_structure_fraction(),
            "charge_at_pH_7": analyzer.charge_at_pH(7.0)  # Added overall charge
        }
    except Exception as e:
        print(f"Error computing properties for sequence: {e}")
        properties = {
            "molecular_weight": 0.0,
            "aromaticity": 0.0,
            "instability_index": 0.0,
            "isoelectric_point": 0.0,
            "gravy": 0.0,
            "secondary_structure_fraction": 0.0,
            "charge_at_pH_7": 0.0
        }
    return properties

def compute_aa_composition(sequence):
    """Computes amino acid composition as fractions."""
    aa_count = Counter(sequence)
    total = len(sequence)
    composition = {aa: count / total for aa, count in aa_count.items()}
    return composition

def compute_residue_features(sequence):
    """Computes per-residue features using aaindex values."""
    features = []
    for aa in sequence:
        if aa in hydrophobicity:
            features.append({
                'hydrophobicity': hydrophobicity[aa],
                'polarity': polarity[aa],
                'volume': volume[aa]
            })
        else:
            features.append({
                'hydrophobicity': 0.0,
                'polarity': 0.0,
                'volume': 0.0
            })
    return features

# PFAM Data
def run_hmmscan_and_parse(fasta_path, hmm_path, output_path):
    if not os.path.exists(output_path):
        print(f"Running hmmscan: {output_path}")
        subprocess.run(["hmmscan", "--domtblout", output_path, hmm_path, fasta_path], check=True)
    else:
        print(f"Using existing hmmscan output: {output_path}")
    return parse_hmmscan_table(output_path)

def parse_hmmscan_table(hmmscan_path):
    """Parses hmmscan tabular output to extract domain information."""
    pfam_data = {}
    with open(hmmscan_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 19:
                continue
            target_name = fields[0]
            uniprot_id = fields[3].split("|")[1]
            start = int(fields[17])
            end = int(fields[18])
            description = " ".join(fields[19:]) if len(fields) > 19 else "Unknown"
            domain_evalue = float(fields[6])
            domain_bias = float(fields[8])

            domain_info = {
                "accession": target_name.split(".")[0],
                "description": description,
                "start": start,
                "end": end,
                "evalue": domain_evalue,
                "bias": domain_bias
            }

            if uniprot_id not in pfam_data:
                pfam_data[uniprot_id] = []
            pfam_data[uniprot_id].append(domain_info)
    return pfam_data

# PDB Data
def extract_dbrefs(pdb_path):
    """Extracts DBREF records from a PDB file to map to UniProt IDs."""
    dbrefs = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("DBREF"):
                chain = line[12].strip()
                database = line[26:32].strip()
                accession = line[33:41].strip()
                dbrefs.append({"chain": chain, "database": database, "accession": accession})
    return dbrefs

def parse_single_pdb(pdb_path):
    """
    Parses a single PDB file and returns a list of (uniprot_id, structural_feature) pairs.
    Handles .gz files by decompressing them temporarily.
    Uses KDTree for efficient contact map calculation.
    """
    
    parser = PDBParser(QUIET=True)
    
    # Handle .gz files by decompressing to a temporary file
    if pdb_path.endswith('.gz'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb') as tmp_file:
            with gzip.open(pdb_path, 'rb') as f_in:
                shutil.copyfileobj(f_in, tmp_file)
            tmp_path = tmp_file.name
    else:
        tmp_path = pdb_path
    
    try:
        structure = parser.get_structure(os.path.basename(pdb_path), tmp_path)
        
        # Extract DBREF records to map to UniProt IDs
        dbrefs = extract_dbrefs(tmp_path)
        uniprot_ids = [ref["accession"] for ref in dbrefs if ref["database"] == "UNP"]
        if not uniprot_ids:
            print(f"No UniProt IDs found in DBREF for {pdb_path}")
            return []
        
        # Extract disulfide bonds from SSBOND records
        with open(tmp_path, "r") as f:
            ssbond_lines = [line for line in f if line.startswith("SSBOND")]
        disulfide_bonds = []
        for line in ssbond_lines:
            try:
                chain1 = line[15].strip()
                res1 = int(line[17:22].strip())
                chain2 = line[29].strip()
                res2 = int(line[31:36].strip())
                disulfide_bonds.append({"chain1": chain1, "res1": res1, "chain2": chain2, "res2": res2})
            except ValueError:
                print(f"Error parsing SSBOND line in {pdb_path}: {line.strip()}")
                continue
        
        # Compute DSSP features (includes solvent accessibility)
        try:
            dssp = DSSP(structure[0], tmp_path)
            residue_details = []
            helix_count = sheet_count = total_residues = 0
            for res in dssp:
                total_residues += 1
                ss = res[2]
                if ss == 'H':
                    helix_count += 1
                elif ss == 'E':
                    sheet_count += 1
                residue_details.append({
                    "secondary_structure": ss,
                    "relative_accessibility": res[3],
                    "phi": res[4],
                    "psi": res[5]
                })
        except Exception as e:
            print(f"Error computing DSSP for {pdb_path}: {e}")
            residue_details = []
            helix_count = sheet_count = total_residues = 0
        
        # Compute residue contact map using KDTree
        ca_atoms = [atom for atom in structure.get_atoms() if atom.get_name() == "CA"]
        ca_coords = np.array([atom.get_coord() for atom in ca_atoms])
        residue_ids = [(atom.get_parent().get_parent().id, atom.get_parent().get_id()[1]) for atom in ca_atoms]
        tree = KDTree(ca_coords)
        pairs = tree.query_pairs(r=CONTACT_THRESHOLD)
        contact_map = [[residue_ids[i], residue_ids[j]] for i, j in pairs]
        
        # Extract Cα coordinates
        ca_coordinates = [
            {"chain": atom.get_parent().get_parent().id, "residue": atom.get_parent().get_id()[1],
             "x": float(atom.get_coord()[0]), "y": float(atom.get_coord()[1]), "z": float(atom.get_coord()[2])}
            for atom in ca_atoms
        ]
        
        # Compile structural features
        structural_feature = {
            "pdb_file": os.path.basename(pdb_path),
            "helix_percentage": (helix_count / total_residues) * 100 if total_residues else 0,
            "sheet_percentage": (sheet_count / total_residues) * 100 if total_residues else 0,
            "coil_percentage": ((total_residues - helix_count - sheet_count) / total_residues) * 100 if total_residues else 0,
            "residue_details": residue_details,
            "disulfide_bonds": disulfide_bonds,
            "contact_map": contact_map,
            "ca_coordinates": ca_coordinates
        }
        
        # Return list of (uniprot_id, structural_feature) pairs
        return [(uid, structural_feature) for uid in uniprot_ids]
    except Exception as e:
        print(f"Error processing {pdb_path}: {e}")
        return []
    finally:
        if pdb_path.endswith('.gz'):
            os.remove(tmp_path)


def integrate_data(uniprot_data, pfam_data, pdb_data):
    """
    Integrates data from UniProt, PFAM, and PDB using UniProt ID as the key.
    Adds gelation status based on domains, structure, and sequence properties.
    """
    for uniprot_id in uniprot_data:
        uniprot_data[uniprot_id]["domains"] = pfam_data.get(uniprot_id, [])
        uniprot_data[uniprot_id]["structural_features"] = pdb_data.get(uniprot_id, [])

        # Determine gelation status
        gelation_from_domains = any(
            domain["accession"] in GELATION_DOMAINS
            for domain in uniprot_data[uniprot_id].get("domains", [])
        )
        gelation_from_structure = any(
            feature["sheet_percentage"] > BETA_SHEET_THRESHOLD
            for feature in uniprot_data[uniprot_id].get("structural_features", [])
        )
        properties = uniprot_data[uniprot_id].get("physicochemical_properties", {})
        gelation_from_sequence = (
            properties.get("gravy", float('-inf')) > GRAVY_THRESHOLD and
            properties.get("instability_index", float('-inf')) > INSTABILITY_THRESHOLD
        )
        uniprot_data[uniprot_id]["gelation"] = "yes" if (
            gelation_from_domains or
            gelation_from_structure or
            gelation_from_sequence
        ) else "no"
    return uniprot_data

def main():
    # Paths to files and directories
    UNIPROT_FASTA_PATH = "../../data/sequence_databases/uniprot/uniprotkb_viridiplantae.txt"
    HMM_PATH = "../../data/sequence_databases/pfam/Pfam-A.hmm"
    HMMSCAN_OUTPUT = "../../data/sequence_databases/pfam/pfam_results.tbl"
    PDB_FOLDER_PATH = "../../data/structure_databases/pdb_data/"
    ALPHAFOLD_PATH = "../../data/structure_databases/alphafold_pdb"

    # Parse UniProt data
    print("Parsing UniProt data (line by line)...")
    uniprot_data = parse_swissprot_file_line_by_line(UNIPROT_FASTA_PATH)

    # Run hmmscan and parse PFAM data
    print("Running hmmscan and parsing PFAM data...")
    pfam_data = run_hmmscan_and_parse(UNIPROT_FASTA_PATH, HMM_PATH, HMMSCAN_OUTPUT)

    # Parse experimental PDB files
    print("Parsing experimental PDB files in parallel...")
    exp_pdb_files = [os.path.join(PDB_FOLDER_PATH, f) for f in os.listdir(PDB_FOLDER_PATH) if f.endswith(".pdb")]
    delayed_exp_tasks = [delayed(parse_single_pdb)(pdb_file) for pdb_file in exp_pdb_files]
    exp_results = compute(*delayed_exp_tasks)

    # Aggregate experimental PDB results
    exp_pdb_data = {}
    for result_list in exp_results:
        for uid, feature in result_list:
            if uid not in exp_pdb_data:
                exp_pdb_data[uid] = []
            exp_pdb_data[uid].append(feature)


    # Identify UniProt IDs without experimental structures
    ids_without_exp = [uid for uid in uniprot_data if uid not in exp_pdb_data]

    # Parse AlphaFold PDB files for those IDs
    print("Parsing AlphaFold PDB files in parallel...")
    alphafold_pdb_files = [
        os.path.join(ALPHAFOLD_PATH, f"AF-{uid}-F1-model_v4.pdb.gz")
        for uid in ids_without_exp
        if os.path.exists(os.path.join(ALPHAFOLD_PATH, f"AF-{uid}-F1-model_v4.pdb.gz"))
    ]
    delayed_alphafold_tasks = [delayed(parse_single_pdb)(pdb_file) for pdb_file in alphafold_pdb_files]
    alphafold_results = compute(*delayed_alphafold_tasks)

    # Aggregate AlphaFold PDB results
    alphafold_pdb_data = {}
    for result_list in alphafold_results:
        for uid, feature in result_list:
            if uid not in alphafold_pdb_data:
                alphafold_pdb_data[uid] = []
            alphafold_pdb_data[uid].append(feature)

    # Combine PDB data: prefer experimental, fall back to AlphaFold
    pdb_data = exp_pdb_data.copy()
    for uid, features in alphafold_pdb_data.items():
        if uid not in pdb_data:
            pdb_data[uid] = features

    # Integrate all data
    print("Integrating data...")
    integrated_data = integrate_data(uniprot_data, pfam_data, pdb_data)

    # Save to MongoDB (uncomment to enable)
    print("Saving to Mongo")
    # write(integrated_data)

    # Save integrated data to a JSON file
    print("Saving integrated data...")
    with open("integrated_data.json", "w") as f:
        json.dump(integrated_data, f, indent=4)

    print("Complete")

if __name__ == "__main__":
    main()