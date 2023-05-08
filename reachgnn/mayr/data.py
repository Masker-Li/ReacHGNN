import os
import pandas as pd
import numpy as np
import os.path as osp
import sys
import json
from typing import Callable, List, Optional
import re
import shutil

import torch
import torch.nn.functional as F
from torch_scatter import scatter
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from torch_geometric.data import (
    HeteroData, InMemoryDataset, download_url, extract_zip)
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from joblib import Parallel, delayed
import torch_geometric.transforms as T

def GetCanonicalMol(mol):
    mol_neworder = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol))], reverse=True)))[1]
    mol_renum = Chem.RenumberAtoms(mol, mol_neworder)
    return mol_renum, mol_neworder

class Base_Reaction_HeteroGraph_DataSet(InMemoryDataset):
    _urls = "XXXXXX"  # 'https://ndownloader.figshare.com/files/3195404'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        # self.atom_types = ['H', 'C', 'N', 'O']
        # self.charge_types = [-1, 0, 1, 2]
        # self.degree_types = [1, 2, 3, 4, 0]
        # self.hybridization_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'S']
        # self.hydrogen_types = [1, 2, 3, 0]
        # self.valence_types = [3, 4, 5, 6, 7, 8]
        # self.bond_types = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'DATIVE']

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def mean(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].mean())

    def std(self, target: int) -> float:
        y = torch.cat([self.get(i).y for i in range(len(self))], dim=0)
        return float(y[:, target].std())

    @property
    def raw_file_names(self) -> List[str]:
        # return ['Radical_ChemSelML/Canonicalized_Radical_ChemSelML_input_data.csv', 'Radical_ChemSelML/sdf/']
        raise NotImplementedError(
            'please set the name of raw files (["*/*.csv", "*/sdf/"]) which will be used in %s.' % self.raw_dir)

    @property
    def processed_file_names(self) -> str:
        # return 'Radical_ChemSelML_Reaction_HeteroGraph_DataSet_MMFF94.pt'
        raise NotImplementedError(
            'please set the name of saved file which will be generated in %s' % self.processed_dir)

    def download(self):
        # pass
        raise NotImplementedError(
            'please download and unzip dataset from %s, and put it at %s' % (self._urls, self.raw_dir))

    @staticmethod
    def parse_smiles_with_MMFF94(smi, sdf_path=None, randomSeed=42):
        if os.path.isfile(sdf_path):
            #rdkitmol = Chem.MolFromMolFile(sdf_path, removeHs=False)
            rdkitmol = Chem.MolFromMolFile(sdf_path)
            stero = Chem.FindPotentialStereo(rdkitmol)
            for element in stero:
                if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                    rdkitmol.GetAtomWithIdx(element.centeredOn).SetProp(
                        'Chirality', str(element.descriptor))
                elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                    rdkitmol.GetBondWithIdx(element.centeredOn).SetProp(
                        'Stereochemistry', str(element.descriptor))
            rdkitmol, _ = GetCanonicalMol(rdkitmol)
            return rdkitmol
        rdkitmol = Chem.MolFromSmiles(smi)
        if not rdkitmol:
            rdkitmol = Chem.MolFromSmiles(smi, sanitize=False)
            print("=====Warning=====", smi, "sanitize=False")
        rdkitmol = Chem.AddHs(rdkitmol)
        rdkitmol.SetProp("_Name", 'Blank '+smi)
        emb_flag = AllChem.EmbedMolecule(
            rdkitmol, maxAttempts=10000, randomSeed=randomSeed)
        if emb_flag != 0:
            print("Something wrong (gen3D) with SMILES: %s, sdf_path: %s" %(smi, sdf_path))
        else:
            rdkitmol.SetProp("_Name", 'Embed  '+smi)
        opt_flag = AllChem.MMFFOptimizeMolecule(rdkitmol, maxIters=10000)
        # print(idx)
        if opt_flag != 0:
            print("Something wrong (opt) with SMILES: %s, sdf_path: %s" %(smi, sdf_path))
        else:
            rdkitmol.SetProp("_Name", 'MMFF94 '+smi)
        conf = rdkitmol.GetConformer()
        if conf:
            Chem.MolToMolFile(rdkitmol, sdf_path)
        rdkitmol = Chem.RemoveHs(rdkitmol)
        stero = Chem.FindPotentialStereo(rdkitmol)
        for element in stero:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                rdkitmol.GetAtomWithIdx(element.centeredOn).SetProp(
                    'Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                rdkitmol.GetBondWithIdx(element.centeredOn).SetProp(
                    'Stereochemistry', str(element.descriptor))
        rdkitmol, _ = GetCanonicalMol(rdkitmol)
        return rdkitmol

    def get_mol(self, smi, sdf_path, maxIters=10):
        for seed in range(maxIters):
            rdkitmol = self.parse_smiles_with_MMFF94(smi, sdf_path, seed)
            try:
                rdkitmol = self.parse_smiles_with_MMFF94(smi, sdf_path, seed)
                if list(rdkitmol.GetConformers()):
                    return rdkitmol
                rdkitmol.GetConformer(0)  # used to raise Error
            except:
                print('\n=====Try another %d times!=====' % seed)
                continue

    @staticmethod
    def get_hetro_index(node_type_x1, node_type_x2, key_site_1=None, key_site_2=None):
        device = node_type_x1.device
        if key_site_1:
            row = torch.tensor([key_site_1], dtype=torch.long, device=device)
        else:
            row = torch.arange(node_type_x1.size(
                0), dtype=torch.long, device=device)
        if key_site_2:
            col = torch.tensor([key_site_2], dtype=torch.long, device=device)
        else:
            col = torch.arange(node_type_x2.size(
                0), dtype=torch.long, device=device)

        row_ = row.view(-1, 1).repeat(1, col.size(0)).view(-1)
        col_ = col.repeat(row.size(0))
        edge_index = torch.stack([row_, col_], dim=0)
        return edge_index

    def mol_to_graph(self, mol, mol_name, mol_dict={}):
        def _chirality(atom):
            if atom.HasProp('Chirality'):
                c_list = [(atom.GetProp('Chirality') == 'Tet_CW'),
                          (atom.GetProp('Chirality') == 'Tet_CCW')]
            else:
                c_list = [0, 0]
            return c_list

        def _stereochemistry(bond):
            if bond.HasProp('Stereochemistry'):
                s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'),
                          (bond.GetProp('Stereochemistry') == 'Bond_Trans')]
            else:
                s_list = [0, 0]
            return s_list

        fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

        if mol_name in mol_dict.keys():
            return mol_dict[mol_name], mol_dict

        N = mol.GetNumAtoms()
        conf = mol.GetConformer()
        pos = conf.GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)

        try:
            feats = chem_feature_factory.GetFeaturesForMol(mol)
        except:
            feats = None
        D_node_list, A_node_list = [], []
        if feats:
            for fi in range(len(feats)):
                if feats[fi].GetFamily() == 'Donor':
                    D_node_list = feats[fi].GetAtomIds()

                elif feats[fi].GetFamily() == 'Acceptor':
                    A_node_list = feats[fi].GetAtomIds()

        type_idx = []
        charge = []
        degree = []
        hybridization = []
        hydrogen = []
        valence = []

        atomic_number = []
        aromatic = []
        isInRing = []
        acceptor = []
        donor = []
        chirality_0, chirality_1 = [], []
        num_hs = []
        for atom in mol.GetAtoms():
            a_type = atom.GetSymbol()
            if a_type not in self.atom_types:
                self.atom_types.append(a_type)
            type_idx.append(self.atom_types.index(a_type))

            a_chg = atom.GetFormalCharge()
            if a_chg not in self.charge_types:
                self.charge_types.append(a_chg)
            charge.append(self.charge_types.index(a_chg))

            a_degree = atom.GetDegree()
            if a_degree not in self.degree_types:
                self.degree_types.append(a_degree)
            degree.append(self.degree_types.index(a_degree))

            a_hybridization = str(atom.GetHybridization())
            if a_hybridization not in self.hybridization_types:
                self.hybridization_types.append(a_hybridization)
            hybridization.append(
                self.hybridization_types.index(a_hybridization))

            a_hydrogen = atom.GetTotalNumHs(includeNeighbors=True)
            if a_hydrogen not in self.hydrogen_types:
                self.hydrogen_types.append(a_hydrogen)
            hydrogen.append(self.hydrogen_types.index(a_hydrogen))

            a_valence = atom.GetTotalValence()
            if a_valence not in self.valence_types:
                self.valence_types.append(a_valence)
            valence.append(self.valence_types.index(a_valence))

            atomic_number.append(atom.GetAtomicNum())
            aromatic.append(1 if atom.GetIsAromatic() else 0)
            isInRing.append(1 if atom.IsInRing() else 0)
            acceptor.append(1 if atom.GetIdx() in A_node_list else 0)
            donor.append(1 if atom.GetIdx() in D_node_list else 0)
            a_stero = _chirality(atom)
            chirality_0.append(a_stero[0])
            chirality_1.append(a_stero[1])

        z = torch.tensor(atomic_number, dtype=torch.long)

        row, col, edge_type = [], [], []
        b_isinring, b_isConjugated = [], []
        b_s0, b_s1 = [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row += [start, end]
            col += [end, start]
            bond_type = str(bond.GetBondType())
            if bond_type not in self.bond_types:
                self.bond_types.append(bond_type)
                #print('warning!!! new bond types: ', bond_type)
            edge_type += 2 * [self.bond_types.index(bond_type)]
            b_isinring += 2*[1 if bond.IsInRing() else 0]
            b_isConjugated += 2*[1 if bond.GetIsConjugated() else 0]
            b_stero = _stereochemistry(bond)
            b_s0 += [b_stero[0], b_stero[0]]
            b_s1 += [b_stero[1], b_stero[1]]

        edge_index = torch.tensor([row, col], dtype=torch.long)
        edge_attr =  torch.tensor([edge_type, b_isinring, b_isConjugated, b_s0, b_s1],
                         dtype=torch.float).t().contiguous()
        
        perm = (edge_index[0] * N + edge_index[1]).argsort()
        edge_index = edge_index[:, perm]
        edge_attr = edge_attr[perm]

        row, col = edge_index
        hs = (z == 1).to(torch.float)
        num_hs = scatter(hs[row], col, dim_size=N).tolist()

        x = torch.tensor([type_idx, charge, degree, hybridization, hydrogen, valence, atomic_number, 
                          aromatic, isInRing, acceptor, donor, chirality_0, chirality_1, num_hs],
                         dtype=torch.float).t().contiguous()

        kwargs = {'x': x, 'z': z, 'pos': pos, 'edge_index': edge_index, 
                  'edge_attr': edge_attr, 'name': int(re.findall(r'\d+', mol_name.split('_-')[-1])[0])}

        mol_dict[mol_name] = kwargs
        return kwargs, mol_dict

    def pre_processe_df(self, df):
        rxn_dict = {}
        cat_col, cat_site, reac_site_col = None, None, None
        for _name in df.columns:
            if 'React_sites' in _name:
                reac_site_col = _name
            elif '_name' in _name:
                _smi = _name.replace('_name', '_smi')
                if _smi in df.columns:
                    if 'Catalyst' in _name or 'catalyst' in _name:
                        cat_col = _name
                    mol_names, _idxs = np.unique(
                        df[_name].values, return_index=True)
                    mol_smis = df[_smi].take(_idxs)
                    mol_lib = Parallel(n_jobs=200, prefer="threads")(
                        delayed(self.get_mol)(smi, self.raw_paths[1]+name+'.sdf') for smi, name in zip(mol_smis, mol_names))
                    mol_dict = {}
                    for mol, mol_name in zip(mol_lib, mol_names):
                        kwargs, mol_dict = self.mol_to_graph(
                            mol, mol_name, mol_dict)

                    rxn_dict[_name] = mol_dict
                    print(_name, 'done!')
            elif cat_col and ("P_site" in _name or "Pd_site" in _name):
                cat_site = _name
        return rxn_dict, reac_site_col, cat_col, cat_site

    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        df = pd.read_csv(self.raw_paths[0], index_col=0)  # .iloc[:200,:]

        data_list = []
        rxn_dict, reac_site_col, cat_col, cat_site = self.pre_processe_df(df)

        print('\nnode_attr and edge_attr list: ')
        print('            self.atom_types = ', self.atom_types)
        print('            self.charge_types = ', sorted(self.charge_types))
        print('            self.degree_types = ', sorted(self.degree_types))
        print('            self.hybridization_types = ', self.hybridization_types)
        print('            self.hydrogen_types = ', sorted(self.hydrogen_types))
        print('            self.valence_types = ', sorted(self.valence_types))
        print('            self.bond_types = ', self.bond_types)
        
        for i in tqdm(df.index):
            #i = i_ + 530

            data = HeteroData()
            for _name in rxn_dict.keys():
                if _name not in df.columns:
                    continue
                mol_dict = rxn_dict[_name]
                kwargs = mol_dict[df[_name][i]]
                
                type_idx = F.one_hot(kwargs['x'][:, 0].long(), num_classes=len(self.atom_types)).float()
                charge = F.one_hot(kwargs['x'][:, 1].long(), num_classes=len(self.charge_types)).float()
                degree = F.one_hot(kwargs['x'][:, 2].long(), num_classes=len(self.degree_types)).float()
                hybridization = F.one_hot(kwargs['x'][:, 3].long(), num_classes=len(self.hybridization_types)).float()
                hydrogen = F.one_hot(kwargs['x'][:, 4].long(), num_classes=len(self.hydrogen_types)).float()
                valence = F.one_hot(kwargs['x'][:, 5].long(), num_classes=len(self.valence_types)).float()
                x = torch.cat([type_idx, charge, degree, hybridization, hydrogen, valence, kwargs['x'][:, 6:]], dim=-1)

                _k = _name.replace('_name', '')
                data[_k].x = x
                data[_k].z = kwargs['z']
                data[_k].pos = kwargs['pos']
                
                edge_attr = kwargs['edge_attr']
                edge_type = F.one_hot(edge_attr[:, 0].long(), num_classes=len(self.bond_types)).float()
                edge_attr = torch.cat([edge_type, edge_attr[:, 1:]], dim=-1)
                data[_k, 'bond', _k].edge_index = kwargs['edge_index']
                data[_k, 'bond', _k].edge_attr = edge_attr
                data[_k].name = kwargs['name']

            react1_col, react2_col = list(rxn_dict.keys())[:2]
            _k1 = react1_col.replace('_name', '')
            _k2 = react2_col.replace('_name', '')
            data[_k1, 'to', _k2].edge_index = self.get_hetro_index(
                data[_k1].x, data[_k2].x)
            data[_k2, 'rev_to', _k1].edge_index = self.get_hetro_index(
                data[_k2].x, data[_k1].x)
            if cat_col:
                _k3 = cat_col.replace('_name', '')
                data[_k1, 'to', _k3].edge_index = self.get_hetro_index(
                    data[_k1].x, data[_k3].x)
                data[_k3, 'rev_to', _k1].edge_index = self.get_hetro_index(
                    data[_k3].x, data[_k1].x)
                data[_k2, 'to', _k3].edge_index = self.get_hetro_index(
                    data[_k2].x, data[_k3].x)
                data[_k3, 'rev_to', _k2].edge_index = self.get_hetro_index(
                    data[_k3].x, data[_k2].x)
                #if cat_site:
                #    key_site = df[cat_site][i]
                #    data[_k1, 'rev_cat', _k3].edge_index = self.get_hetro_index(
                #        data[_k1].x, data[_k3].x, key_site_2=key_site)
                #    data[_k3, 'cat', _k1].edge_index = self.get_hetro_index(
                #        data[_k3].x, data[_k1].x, key_site_1=key_site)
                #    data[_k2, 'rev_cat', _k3].edge_index = self.get_hetro_index(
                #        data[_k2].x, data[_k3].x, key_site_2=key_site)
                #    data[_k3, 'cat', _k2].edge_index = self.get_hetro_index(
                #        data[_k3].x, data[_k2].x, key_site_1=key_site)

            y = torch.tensor(df['Output'][i]).unsqueeze(0)
            _sites = eval(df[reac_site_col][i])
            _sites = [_sites[0][0], _sites[1][0]]
            reac_site = torch.tensor(_sites).flatten().unsqueeze(0)
            _sites = eval(df['Prod_sites'][i])
            _sites = [_sites[0][0], _sites[1][0]]
            prod_site = torch.tensor(_sites).flatten().unsqueeze(0)

            data['rxn'].y = y
            data['rxn'].idx = i
            data['rxn'].reac_site = reac_site
            data['rxn'].prod_site = reac_site

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
        
class Mayr_HeteroGraph_DataSet(Base_Reaction_HeteroGraph_DataSet):
    _urls = "XXXXXX"  # 'https://ndownloader.figshare.com/files/3195404'
    
    def __init__(self, root: str, postfix: str = 'train', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        if True:
            self.atom_types = []
            self.charge_types = []
            self.degree_types = []
            self.hybridization_types = []
            self.hydrogen_types = []
            self.valence_types = []
            self.bond_types = []
            self.bond_types = []
        if True:
            self.atom_types =  ['H', 'C', 'N', 'O', 'F', 'S', 'Si', 'Ge', 'Sn', 'Cl', 'B', 'Hg', 'Pb', 'Br', 'I', 'P']
            self.charge_types =  [-1, 0, 1]
            self.degree_types =  [0, 1, 2, 3, 4, 6]
            self.hybridization_types =  ['S', 'SP', 'SP2', 'SP3', 'SP3D2']
            self.hydrogen_types =  [0, 1, 2, 3, 4]
            self.valence_types =  [0, 1, 2, 3, 4, 5, 6, 7]
            self.bond_types =  ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
            
        self.postfix = postfix
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['Mayr/Canonicalized_Mayr_input_data_%s.csv'%self.postfix, 'Mayr/sdf/']

    @property
    def processed_file_names(self) -> str:
        return 'Mayr_HeteroGraph_DataSet_MMFF94_%s.pt'%self.postfix

    def download(self):
        pass
        # raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (self._urls, self.raw_dir))
        
    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if not os.path.isdir(self.raw_paths[-1]):
            os.makedirs(self.raw_paths[-1])

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        df = pd.read_csv(self.raw_paths[0], index_col=0)  # .iloc[:200,:]

        data_list = []
        rxn_dict, reac_site_col, cat_col, cat_site = self.pre_processe_df(df)

        print('\nnode_attr and edge_attr list: ')
        print('            self.atom_types = ', self.atom_types)
        print('            self.charge_types = ', sorted(self.charge_types))
        print('            self.degree_types = ', sorted(self.degree_types))
        print('            self.hybridization_types = ', self.hybridization_types)
        print('            self.hydrogen_types = ', sorted(self.hydrogen_types))
        print('            self.valence_types = ', sorted(self.valence_types))
        print('            self.bond_types = ', self.bond_types)
        
        for i in tqdm(df.index):
            #i = i_ + 530

            data = HeteroData()
            for _name in rxn_dict.keys():
                if _name not in df.columns:
                    continue
                mol_dict = rxn_dict[_name]
                kwargs = mol_dict[df[_name][i]]
                
                type_idx = F.one_hot(kwargs['x'][:, 0].long(), num_classes=len(self.atom_types)).float()
                charge = F.one_hot(kwargs['x'][:, 1].long(), num_classes=len(self.charge_types)).float()
                degree = F.one_hot(kwargs['x'][:, 2].long(), num_classes=len(self.degree_types)).float()
                hybridization = F.one_hot(kwargs['x'][:, 3].long(), num_classes=len(self.hybridization_types)).float()
                hydrogen = F.one_hot(kwargs['x'][:, 4].long(), num_classes=len(self.hydrogen_types)).float()
                valence = F.one_hot(kwargs['x'][:, 5].long(), num_classes=len(self.valence_types)).float()
                x = torch.cat([type_idx, charge, degree, hybridization, hydrogen, valence, kwargs['x'][:, 6:]], dim=-1)

                _k = _name.replace('_name', '')
                data[_k].x = x
                data[_k].z = kwargs['z']
                data[_k].pos = kwargs['pos']
                
                edge_attr = kwargs['edge_attr']
                edge_type = F.one_hot(edge_attr[:, 0].long(), num_classes=len(self.bond_types)).float()
                edge_attr = torch.cat([edge_type, edge_attr[:, 1:]], dim=-1)
                data[_k, 'bond', _k].edge_index = kwargs['edge_index']
                data[_k, 'bond', _k].edge_attr = edge_attr
                data[_k].name = kwargs['name']

            react1_col, react2_col, sol_col = list(rxn_dict.keys())[:3]
            _k1 = react1_col.replace('_name', '')
            _k2 = react2_col.replace('_name', '')
            _k3 = sol_col.replace('_name', '')
            data[_k1, 'to', _k2].edge_index = self.get_hetro_index(
                data[_k1].x, data[_k2].x)
            data[_k2, 'rev_to', _k1].edge_index = self.get_hetro_index(
                data[_k2].x, data[_k1].x)
            data[_k1, 'to', _k3].edge_index = self.get_hetro_index(
                data[_k1].x, data[_k3].x)
            data[_k3, 'rev_to', _k1].edge_index = self.get_hetro_index(
                data[_k3].x, data[_k1].x)

            y = torch.tensor(df['Output'][i]).unsqueeze(0)
            #_sites = eval(df[reac_site_col][i])
            #_sites = [_sites[0][0], _sites[1][0]]
            #reac_site = torch.tensor(_sites).flatten().unsqueeze(0)
            #_sites = eval(df['Prod_sites'][i])
            #_sites = [_sites[0][0], _sites[1][0]]
            #prod_site = torch.tensor(_sites).flatten().unsqueeze(0)

            data['rxn'].y = y
            data['rxn'].N = torch.tensor(df['N_parameter'][i]).unsqueeze(0)
            data['rxn'].E = torch.tensor(df['E_parameter'][i]).unsqueeze(0)
            data['rxn'].sN = torch.tensor(df['sN_parameter'][i]).unsqueeze(0)
            data['rxn'].idx = i
            #data['rxn'].reac_site = reac_site
            #data['rxn'].prod_site = reac_site

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])
    
class Mayr_HeteroGraph_DataSet_inference(Base_Reaction_HeteroGraph_DataSet):
    _urls = "XXXXXX"  # 'https://ndownloader.figshare.com/files/3195404'
    
    def __init__(self, root: str, postfix: str = 'train', transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        if True:
            self.atom_types = []
            self.charge_types = []
            self.degree_types = []
            self.hybridization_types = []
            self.hydrogen_types = []
            self.valence_types = []
            self.bond_types = []
            self.bond_types = []
        if True:
            self.atom_types =  ['H', 'C', 'N', 'O', 'F', 'S', 'Si', 'Ge', 'Sn', 'Cl', 'B', 'Hg', 'Pb', 'Br', 'I', 'P']
            self.charge_types =  [-1, 0, 1]
            self.degree_types =  [0, 1, 2, 3, 4]
            self.hybridization_types =  ['S', 'SP', 'SP2', 'SP3']
            self.hydrogen_types =  [0, 1, 2, 3, 4]
            self.valence_types =  [0, 1, 2, 3, 4, 5, 6]
            self.bond_types =  ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
            
        self.postfix = postfix
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ['Mayr/Canonicalized_Batteries_data_%s.csv'%self.postfix, 'Mayr/sdf/']

    @property
    def processed_file_names(self) -> str:
        return 'Mayr_HeteroGraph_DataSet_MMFF94_inference_%s.pt'%self.postfix

    def download(self):
        pass
        # raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (self._urls, self.raw_dir))
        
    def process(self):
        try:
            import rdkit
            from rdkit import Chem, RDLogger
            from rdkit.Chem.rdchem import BondType as BT
            from rdkit.Chem.rdchem import HybridizationType
            RDLogger.DisableLog('rdApp.*')

        except ImportError:
            rdkit = None

        if not os.path.isdir(self.raw_paths[-1]):
            os.makedirs(self.raw_paths[-1])

        if rdkit is None:
            print(("Using a pre-processed version of the dataset. Please "
                   "install 'rdkit' to alternatively process the raw data."),
                  file=sys.stderr)

            data_list = torch.load(self.raw_paths[0])
            data_list = [Data(**data_dict) for data_dict in data_list]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[0])
            return

        df = pd.read_csv(self.raw_paths[0], index_col=0)  # .iloc[:200,:]

        data_list = []
        rxn_dict, reac_site_col, cat_col, cat_site = self.pre_processe_df(df)

        print('\nnode_attr and edge_attr list: ')
        print('            self.atom_types = ', self.atom_types)
        print('            self.charge_types = ', sorted(self.charge_types))
        print('            self.degree_types = ', sorted(self.degree_types))
        print('            self.hybridization_types = ', self.hybridization_types)
        print('            self.hydrogen_types = ', sorted(self.hydrogen_types))
        print('            self.valence_types = ', sorted(self.valence_types))
        print('            self.bond_types = ', self.bond_types)
        
        for i in tqdm(df.index):
            #i = i_ + 530

            data = HeteroData()
            for _name in rxn_dict.keys():
                if _name not in df.columns:
                    continue
                mol_dict = rxn_dict[_name]
                kwargs = mol_dict[df[_name][i]]
                
                type_idx = F.one_hot(kwargs['x'][:, 0].long(), num_classes=len(self.atom_types)).float()
                charge = F.one_hot(kwargs['x'][:, 1].long(), num_classes=len(self.charge_types)).float()
                degree = F.one_hot(kwargs['x'][:, 2].long(), num_classes=len(self.degree_types)).float()
                hybridization = F.one_hot(kwargs['x'][:, 3].long(), num_classes=len(self.hybridization_types)).float()
                hydrogen = F.one_hot(kwargs['x'][:, 4].long(), num_classes=len(self.hydrogen_types)).float()
                valence = F.one_hot(kwargs['x'][:, 5].long(), num_classes=len(self.valence_types)).float()
                x = torch.cat([type_idx, charge, degree, hybridization, hydrogen, valence, kwargs['x'][:, 6:]], dim=-1)

                _k = _name.replace('_name', '')
                data[_k].x = x
                data[_k].z = kwargs['z']
                data[_k].pos = kwargs['pos']
                
                edge_attr = kwargs['edge_attr']
                edge_type = F.one_hot(edge_attr[:, 0].long(), num_classes=len(self.bond_types)).float()
                edge_attr = torch.cat([edge_type, edge_attr[:, 1:]], dim=-1)
                data[_k, 'bond', _k].edge_index = kwargs['edge_index']
                data[_k, 'bond', _k].edge_attr = edge_attr
                data[_k].name = kwargs['name']

            react1_col, react2_col, sol_col = list(rxn_dict.keys())[:3]
            _k1 = react1_col.replace('_name', '')
            _k2 = react2_col.replace('_name', '')
            _k3 = sol_col.replace('_name', '')
            data[_k1, 'to', _k2].edge_index = self.get_hetro_index(
                data[_k1].x, data[_k2].x)
            data[_k2, 'rev_to', _k1].edge_index = self.get_hetro_index(
                data[_k2].x, data[_k1].x)
            data[_k1, 'to', _k3].edge_index = self.get_hetro_index(
                data[_k1].x, data[_k3].x)
            data[_k3, 'rev_to', _k1].edge_index = self.get_hetro_index(
                data[_k3].x, data[_k1].x)

            y = torch.tensor(df['Output'][i]).unsqueeze(0)
            #_sites = eval(df[reac_site_col][i])
            #_sites = [_sites[0][0], _sites[1][0]]
            #reac_site = torch.tensor(_sites).flatten().unsqueeze(0)
            #_sites = eval(df['Prod_sites'][i])
            #_sites = [_sites[0][0], _sites[1][0]]
            #prod_site = torch.tensor(_sites).flatten().unsqueeze(0)

            data['rxn'].y = y
            data['rxn'].N = torch.tensor(df['N_parameter'][i]).unsqueeze(0)
            data['rxn'].E = torch.tensor(df['E_parameter'][i]).unsqueeze(0)
            data['rxn'].sN = torch.tensor(df['sN_parameter'][i]).unsqueeze(0)
            data['rxn'].idx = i
            #data['rxn'].reac_site = reac_site
            #data['rxn'].prod_site = reac_site

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])

from itertools import product
from sklearn.utils import shuffle


def DataSplit(root=r'data/', batch_size=32):
    dataset_train = Mayr_HeteroGraph_DataSet(root, postfix='train')
    dataset_val = Mayr_HeteroGraph_DataSet(root, postfix='val0')
    dataset_val1 = Mayr_HeteroGraph_DataSet(root, postfix='val1')
    dataset_val2 = Mayr_HeteroGraph_DataSet(root, postfix='val2')
    dataset_test = Mayr_HeteroGraph_DataSet(root, postfix='test0')
    dataset_test1 = Mayr_HeteroGraph_DataSet(root, postfix='test1')
    dataset_test2 = Mayr_HeteroGraph_DataSet(root, postfix='test2')

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    val_loader1 = DataLoader(dataset_val1, batch_size=batch_size, shuffle=False)
    val_loader2 = DataLoader(dataset_val2, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    test_loader1 = DataLoader(dataset_test1, batch_size=batch_size, shuffle=False)
    test_loader2 = DataLoader(dataset_test2, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_loader1, val_loader2, test_loader, test_loader1, test_loader2
