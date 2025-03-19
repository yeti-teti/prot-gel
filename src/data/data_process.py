from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from Bio import AlignIO

from Bio.PDB import PDBParser

import gzip

## Sequence database

# UniProt
# sequences = list(SeqIO.parse("../../datasets/sequence_databases/uniprot/uniprot_plant.fasta", "fasta"))
# # Computing physicochemical properties
# for seq_record in sequences:
#     analysis = ProteinAnalysis(str(seq_record.seq))
#     hydrophobicity = analysis.gravy()


# PFAM
pfam_file = "../../datasets/sequence_databases/pfam/Pfam-A.seed"
alignments = []
for alignment in AlignIO.parse(pfam_file, "stockholm"):
    alignments.append(alignment)

# Now 'alignments' is a list of MultipleSeqAlignment objects
print(f"Number of alignments found: {len(alignments)}")
# for i, align in enumerate(alignments):
#     print(f"Alignment {i + 1}: {len(align)} sequences")


# UniRef90
# uniref_sequences = list(SeqIO.parse("../../datasets/sequence_databases/uniprot/uniref90_plant.fasta", "fasta"))


## Protein Structure Databases

# PDB
parser = PDBParser()
structure = parser.get_structure("protein", "../../datasets/structure_databases/pdb_data/1A2S.pdb")


## Genome Sequence Databases

# JGI Phytozome
with gzip.open("../../datasets/genome_sequence_databases/phytozome/Phytozome/Alinifolium_472_v1.1.protein.fa.gz", "rt") as handle:
    sequences = list(SeqIO.parse(handle, "fasta"))

