import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Crippen, Lipinski, QED
from rdkit.Chem import rdMolDescriptors
from sascorer import calculateScore
# from analysis.molecule_builder import get_bond_order_batch, build_molecule
# from constants import allowed_bonds
from espsim.electrostatics import GetMolProps, GetEspSim
from rdkit.Chem import AllChem, rdShapeHelpers
from rdkit.Chem.FeatMaps import FeatMaps
from rdkit import RDConfig

import os.path as osp

def read_sdf(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file)
    mols_list = [i for i in supp]
    return mols_list

class MoleculeProperties:
    '''
    Description:
    calculate the properties of signle molecle:
        functions:
        calculate_qed: calculate the QED (quantitative estimate of drug-likeness) of the molecule
        calculate_sa: calculate the synthetic accessibility of the molecule
        calculate_logp: calculate the logP of the molecule
        calculate_lipinski: calculate the lipinski of the molecule
        calculate_sasa: calculate the solvent accessible surface area of the molecule
        calculate_tpsa: calculate the topological polar surface area of the molecule
        calculate_mw: calculate the molecular weight of the molecule
        calculate_hba: calculate the number of hydrogen bond acceptors of the molecule
        calculate_hbd: calculate the number of hydrogen bond donors of the molecule

    Calculate the properties of batch molecules:
        functions:
        calculate_similarity (2D): calculate the similarity between a molecule and b molecule 
        calculate_esp: (3D) calculate the electrostatic similarity between the molecule and the target
        calculate_shape: (3D) calculate the shape similarity between the molecule and the target
        calculate_rc_score: (3D) calculate the pharmacophore score combined with shape of the molecule

        calculate_diversity: calculate the diversity of a batch molecules
    '''
    def __init__(self):
        fdefName = 'BaseFeatures.fdef'
        if not osp.exists(fdefName):
            fdefName =  os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        self.fdef = AllChem.BuildFeatureFactory(fdefName)
        self.fmParams = {}
        for k in self.fdef.GetFeatureFamilies():
            fparams = FeatMaps.FeatMapParams()
            self.fmParams[k] = fparams
        self.keep = ('Donor', 'Acceptor', 'NegIonizable', 'PosIonizable', 
        'ZnBinder', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe')
    
    @staticmethod
    def calculate_qed(rdmol):
        return QED.qed(rdmol)

    @staticmethod
    def calculate_sa(rdmol):
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)  # from pocket2mol

    @staticmethod
    def calculate_logp(rdmol):
        return Crippen.MolLogP(rdmol)

    @staticmethod
    def calculate_lipinski(rdmol):
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

    def calculate_sasa(self, mol):
        pdb_file = './tmp.pdb'
        Chem.MolToPDBFile(mol, pdb_file)
        sasa = self.sasa_(pdb_file)
        os.remove(pdb_file)
        return sasa
    
    def sasa_(self, pdb_file):
        import mdtraj as md
        # Load the PDB file
        traj = md.load(pdb_file)
        # Compute the SASA
        sasa = md.shrake_rupley(traj, mode='residue')
        # Get the total SASA
        total_sasa = np.sum(sasa)

        return total_sasa
    
    @staticmethod
    def hbd(mol):
        return rdMolDescriptors.CalcNumHBA(mol)

    @staticmethod
    def hba(mol):
        return rdMolDescriptors.CalcNumHBD(mol)
    
    @staticmethod
    def tpsa(mol):
        return rdMolDescriptors.CalcTPSA(mol)

    @staticmethod
    def calculate_mw(mol):
        return Descriptors.MolWt(mol)
    
    @staticmethod
    def calculate_similarity(mol_a, mol_b):
        fp1 = Chem.RDKFingerprint(mol_a)
        fp2 = Chem.RDKFingerprint(mol_b)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    @staticmethod
    def calculate_esp(mol_a, mol_b):
        return GetEspSim(mol_a, mol_b)
    @staticmethod
    def calculate_shape(mol_a, mol_b):
        return 1-AllChem.ShapeTanimotoDist(mol_a, mol_b)
    
    def get_FeatureMapScore(self, query_mol, ref_mol):
        featLists = []
        for m in [query_mol, ref_mol]:
            rawFeats = self.fdef.GetFeaturesForMol(m)
            # filter that list down to only include the ones we're intereted in
            featLists.append([f for f in rawFeats if f.GetFamily() in self.keep])
        fms = [FeatMaps.FeatMap(feats=x, weights=[1] * len(x), params=self.fmParams) for x in featLists]
        fms[0].scoreMode = FeatMaps.FeatMapScoreMode.Best
        fm_score = fms[0].ScoreFeats(featLists[1]) / min(fms[0].GetNumFeatures(), len(featLists[1]))
        return fm_score
    
    def calculate_sc_score(self, query_mol, ref_mol):
        fm_score = self.get_FeatureMapScore(query_mol, ref_mol)

        protrude_dist = rdShapeHelpers.ShapeProtrudeDist(query_mol, ref_mol,
                allowReordering=False)
        SC_RDKit_score = 0.5*fm_score + 0.5*(1 - protrude_dist)
        return SC_RDKit_score

    @classmethod
    def calculate_diversity(cls, batch_mols):
        if len(batch_mols) < 2:
            return 0.0

        div = 0
        total = 0
        for i in range(len(batch_mols)):
            for j in range(i + 1, len(batch_mols)):
                div += 1 - cls.similarity(batch_mols[i], batch_mols[j])
                total += 1

        return div / total

if __name__ == '__main__':
    property_computer = MoleculeProperties()
    mol = Chem.MolFromSmiles('CCO')
    print(property_computer.calculate_sasa(mol))