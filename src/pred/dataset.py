import torch
from torch.utils.data import Dataset
import numpy as np
import json

class ProteinDataset(Dataset):
    def __init__(self, json_path, is_train=True, mean_std=None):
        """Initialize dataset from JSON file."""
        with open(json_path, 'r') as f:
            self.data = list(json.load(f).values())  # Convert dict to list of values
        self.is_train = is_train
        self.ss_classes = ['H', 'G', 'I', 'E', 'B', 'T', 'S', '-']  # DSSP secondary structure classes
        self.aa_to_int = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}  # 20 amino acids
        self.gelation_domains = ["PF00190", "PF04702", "PF00234"]
        
        if is_train:
            self.mean_std = self.compute_mean_std()
        else:
            if mean_std is None:
                raise ValueError("mean_std must be provided for test dataset")
            self.mean_std = mean_std

    def compute_mean_std(self):
        """Compute mean and std for continuous features across the dataset."""
        all_residue_features = []
        all_protein_features = []

        for protein in self.data:
            # Residue features
            for i in range(len(protein['sequence'])):
                res_feat = protein['residue_features'][i]
                hydro = res_feat['hydrophobicity']
                polar = res_feat['polarity']
                vol = res_feat['volume']
                if 'structural_features' in protein and protein['structural_features']:
                    struct = protein['structural_features'][0]
                    if i < len(struct['residue_details']):
                        res_detail = struct['residue_details'][i]
                        acc = res_detail['relative_accessibility']
                        phi = res_detail['phi']
                        psi = res_detail['psi']
                    else:
                        acc = phi = psi = 0.0
                else:
                    acc = phi = psi = 0.0
                all_residue_features.append([hydro, polar, vol, acc, phi, psi])
            
            # Protein features
            phy_prop = protein['physicochemical_properties']
            struct = protein['structural_features'][0] if 'structural_features' in protein and protein['structural_features'] else {
                'helix_percentage': 0, 'sheet_percentage': 0, 'coil_percentage': 0
            }
            features = [
                len(protein['sequence']),
                phy_prop['molecular_weight'],
                phy_prop['aromaticity'],
                phy_prop['instability_index'],
                phy_prop['isoelectric_point'],
                phy_prop['gravy'],
                phy_prop['charge_at_pH_7'],
                struct['helix_percentage'],
                struct['sheet_percentage'],
                struct['coil_percentage']
            ]
            domains = protein['domains']
            for gd in self.gelation_domains:
                features.append(1 if any(d['accession'] == gd for d in domains) else 0)
            all_protein_features.append(features)

        # Compute statistics
        residue_array = np.array(all_residue_features)
        protein_array = np.array(all_protein_features)
        return {
            'residue_mean': residue_array.mean(axis=0),
            'residue_std': residue_array.std(axis=0) + 1e-6,
            'protein_mean': protein_array.mean(axis=0),
            'protein_std': protein_array.std(axis=0) + 1e-6
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return processed features for a single protein."""
        protein = self.data[idx]
        sequence = protein['sequence']
        
        # Encode sequence
        seq_encoded = [self.aa_to_int.get(aa, 20) for aa in sequence]  # 20 for unknown
        
        # Residue features
        residue_features_list = []
        for i in range(len(sequence)):
            res_feat = protein['residue_features'][i]
            hydro = res_feat['hydrophobicity']
            polar = res_feat['polarity']
            vol = res_feat['volume']
            if 'structural_features' in protein and protein['structural_features']:
                struct = protein['structural_features'][0]
                if i < len(struct['residue_details']):
                    res_detail = struct['residue_details'][i]
                    ss = res_detail['secondary_structure']
                    acc = res_detail['relative_accessibility']
                    phi = res_detail['phi']
                    psi = res_detail['psi']
                    ss_onehot = [1 if ss == cls else 0 for cls in self.ss_classes]
                else:
                    ss_onehot = [0] * len(self.ss_classes)
                    acc = phi = psi = 0.0
            else:
                ss_onehot = [0] * len(self.ss_classes)
                acc = phi = psi = 0.0
            
            cont_features = [hydro, polar, vol, acc, phi, psi]
            for j in range(len(cont_features)):
                cont_features[j] = (cont_features[j] - self.mean_std['residue_mean'][j]) / self.mean_std['residue_std'][j]
            residue_features_list.append(cont_features + ss_onehot)
        residue_features = torch.tensor(residue_features_list, dtype=torch.float)

        # Protein features
        phy_prop = protein['physicochemical_properties']
        struct = protein['structural_features'][0] if 'structural_features' in protein and protein['structural_features'] else {
            'helix_percentage': 0, 'sheet_percentage': 0, 'coil_percentage': 0
        }
        protein_features = [
            len(sequence),
            phy_prop['molecular_weight'],
            phy_prop['aromaticity'],
            phy_prop['instability_index'],
            phy_prop['isoelectric_point'],
            phy_prop['gravy'],
            phy_prop['charge_at_pH_7'],
            struct['helix_percentage'],
            struct['sheet_percentage'],
            struct['coil_percentage']
        ]
        domains = protein['domains']
        for gd in self.gelation_domains:
            protein_features.append(1 if any(d['accession'] == gd for d in domains) else 0)
        for j in range(len(protein_features)):
            protein_features[j] = (protein_features[j] - self.mean_std['protein_mean'][j]) / self.mean_std['protein_std'][j]
        protein_features = torch.tensor(protein_features, dtype=torch.float)

        # Label
        gelation = 1 if protein['gelation'] == 'yes' else 0

        return {
            'sequence': torch.tensor(seq_encoded, dtype=torch.long),
            'residue_features': residue_features,
            'protein_features': protein_features,
            'gelation': torch.tensor(gelation, dtype=torch.float)
        }