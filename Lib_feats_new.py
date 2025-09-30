

### Import Sklearn part
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import euclidean_distances

### Import RdKit packages
import rdkit
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

### Import other packages
import numpy as np
import pandas as pd
from collections import defaultdict
import copy
import statistics
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import glob
import re



def sum_over_bonds(mol_list, predefined_bond_types=[], return_names=True):

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]

    empty_bond_dict = defaultdict(lambda : 0)
    num_mols = len(mol_list)

    if (len(predefined_bond_types) == 0 ):
        for i, mol in enumerate(mol_list):
            bonds = mol.GetBonds()
            for bond in bonds:
                bond_start_atom = bond.GetBeginAtom().GetSymbol()
                bond_end_atom = bond.GetEndAtom().GetSymbol()
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                bond_atoms = [bond_start_atom, bond_end_atom]
                if (bond_type == ''):
                    bond_type = "-"
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                empty_bond_dict[bond_string] = 0
    else:
        for bond_string in predefined_bond_types:
            empty_bond_dict[bond_string] = 0

    bond_types = list(empty_bond_dict.keys())
    num_bond_types = len(bond_types)

    X_LBoB = np.zeros([num_mols, num_bond_types])

    for i, mol in enumerate(mol_list):
        bonds = mol.GetBonds()
        bond_dict = copy.deepcopy(empty_bond_dict)
        for bond in bonds:
            bond_start_atom = bond.GetBeginAtom().GetSymbol()
            bond_end_atom = bond.GetEndAtom().GetSymbol()
            
            if (bond_start_atom=='*' or bond_end_atom=='*'):
                pass
            else:
                bond_type = bond.GetSmarts(allBondsExplicit=True)
                if (bond_type == ''):
                    bond_type = "-"
                bond_atoms = [bond_start_atom, bond_end_atom]
                bond_string = min(bond_atoms)+bond_type+max(bond_atoms)
                bond_dict[bond_string] += 1

        
        X_LBoB[i,:] = [bond_dict[bond_type] for bond_type in bond_types]

    if (return_names):
        return bond_types, X_LBoB
    else:
        return X_LBoB

def literal_bag_of_bonds(mol_list, predefined_bond_types=[]):
    return sum_over_bonds(mol_list, predefined_bond_types=predefined_bond_types)


### Linh Estate

def truncated_Estate_featurizer(mol_list, return_names=False):
    
    X = np.array([FingerprintMol(mol)[0][6:37] for mol in mol_list])
    Estate_names=['-CH3', '=CH2', '—CH2—', '\\#CH', '=CH-', 'aCHa', '>CH-', '=c=', '\\#C-', '=C$<$', 'aCa',
    'aaCa', '$>$C$<$', '-NH3[+1]', '-NH2', '-NH2-[+1]', '=NH', '-NH-', 'aNHa', '\\#N', '$>$NH-[+1]',
    '=N—', 'aNa', '$>$N—', '—N$<$$<$', 'aaNs', '$>$N$<$[+1]', '-OH', '=0', '-0-', 'aOa']
    if (return_names == True):
        return Estate_names, X
    else:
        return X




def get_num_atom(mol, atomic_number):
    '''returns the number of atoms of particular atomic number'''
    num = 0
    for atom in mol.GetAtoms():
        atom_num = atom.GetAtomicNum()
        if (atom_num == atomic_number):
            num += 1
    return num

def oxygen_balance_100(mol, scaled=False):
    '''returns the OB_100 descriptor'''
    n_O = get_num_atom(mol, 8)
    n_C = get_num_atom(mol, 6)
    n_H = get_num_atom(mol, 1)
    n_atoms = mol.GetNumAtoms()
    OB_100 = 100*(n_O - 2*n_C - n_H/2)/n_atoms

    if scaled:
        return (OB_100 + 28.5)/18.96 + 1.0 ## rescales to center around one with unit variance based on statistics of the combined energetics dataset
    else:
        return OB_100

def oxygen_balance_1600(mol):
    '''returns the OB_16000 descriptor'''
    n_O = get_num_atom(mol, 8)
    n_C = get_num_atom(mol, 6)
    n_H = get_num_atom(mol, 1)
    mol_weight = Descriptors.ExactMolWt(mol)
    return 1600*(n_O - 2*n_C - n_H/2)/mol_weight


def return_combined_nums_update(mol):
    return custom_descriptor_set_update(mol)


def get_neigh_dict(atom):
    '''returns a dictionary with the number of neighbors for a given atom'''
    neighs = defaultdict(int)
    for atom in atom.GetNeighbors():
        neighs[atom.GetSymbol()] += 1
    return neighs


def get_num_with_neighs(mol, central_atom, target_dict):
    '''returns how many atoms of a particular type have a particular configuration of neighbora'''
    target_num = 0
    for key in list(target_dict.keys()):
        target_num += target_dict[key]

    num = 0
    for atom in mol.GetAtoms():
        if (atom.GetSymbol() == central_atom):
            target = True
            nbs = get_neigh_dict(atom)
            for key in list(target_dict.keys()):
                if (nbs[key] != target_dict[key]):
                    target = False
                    break

            n_nbs = len(atom.GetNeighbors())
            if (target_num != n_nbs):
                target = False

            if (target):
                num +=1

    return num
def custom_descriptor_set(mol):
    n_a = mol.GetNumAtoms()
    n_C = get_num_atom(mol, 6)
    n_N = get_num_atom(mol, 7)
    n_O = get_num_atom(mol, 8)
    n_H = get_num_atom(mol, 1)
    n_F = get_num_atom(mol, 9)
    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})

    if (n_C == 0):
        NCratio = n_N/0.00001
    else:
        NCratio = n_N/n_C

    return [oxygen_balance_100(mol, scaled=True), n_C, n_N, n_O1, n_O2, n_O3, n_O4, n_H, n_F, NCratio,
                get_num_with_neighs(mol, 'N', {'O': 2, 'C': 1}) ,  #CNO2
                get_num_with_neighs(mol, 'N', {'O': 2, 'N': 1}) ,  #NNO2
                get_num_with_neighs(mol, 'N', {'O': 2})         ,  #ONO
                get_num_with_neighs(mol, 'N', {'O': 3})         ,  #ONO2
                get_num_with_neighs(mol, 'N', {'N': 1, 'C': 1}) ,  #CNN
                get_num_with_neighs(mol, 'N', {'N': 2})         ,  #NNN
                get_num_with_neighs(mol, 'N', {'C': 1,'O': 1})  ,  #CNO
                get_num_with_neighs(mol, 'N', {'C': 1,'H': 2})  ,  #CNH2
                get_num_with_neighs(mol, 'N', {'C': 2,'O': 1})  ,  #CN(O)C
                get_num_with_neighs(mol, 'F', {'C': 1})         ,  #CF
                get_num_with_neighs(mol, 'N', {'C': 1, 'N':2})     #CNF
                #n_C/n_a, n_N/n_a, n_O/n_a, n_H/n_a
           ]
def custom_descriptor_set_update(mol):
    n_a = mol.GetNumAtoms()
    n_C = get_num_atom(mol, 6)
    n_N = get_num_atom(mol, 7)
    n_O = get_num_atom(mol, 8)
    n_H = get_num_atom(mol, 1)
    # these 3 following changed to ABOCH
    n_F = get_num_atom(mol, 9)
    ## By knowing that, sevreal data will not have F, use Br and CL instead
    n_Cl = get_num_atom(mol, 17)
    n_Br= get_num_atom(mol, 35)

    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})

    if (n_C == 0):
        NCratio = n_N/0.00001
    else:
        NCratio = n_N/n_C

    return [oxygen_balance_100(mol, scaled=True), n_C, n_N, n_O1, n_O2, n_O3, n_O4, n_H, n_F, n_Cl, n_Br, NCratio,
                get_num_with_neighs(mol, 'N', {'O': 2, 'C': 1}) ,  #CNO2
                get_num_with_neighs(mol, 'N', {'O': 2, 'N': 1}) ,  #NNO2
                get_num_with_neighs(mol, 'N', {'O': 2})         ,  #ONO
                get_num_with_neighs(mol, 'N', {'O': 3})         ,  #ONO2
                get_num_with_neighs(mol, 'N', {'N': 1, 'C': 1}) ,  #CNN
                get_num_with_neighs(mol, 'N', {'N': 2})         ,  #NNN
                get_num_with_neighs(mol, 'N', {'C': 1,'O': 1})  ,  #CNO
                get_num_with_neighs(mol, 'N', {'C': 1,'H': 2})  ,  #CNH2
                get_num_with_neighs(mol, 'N', {'C': 2,'O': 1})  ,  #CN(O)C
                get_num_with_neighs(mol, 'F', {'C': 1})         ,  #CF
                get_num_with_neighs(mol, 'N', {'C': 1, 'N':2})     #CNF
                #n_C/n_a, n_N/n_a, n_O/n_a, n_H/n_a
           ]

def CDS_featurizer(mol_list, return_names=True):
    X_CDS = []
    for mol in mol_list:
        X_CDS += [custom_descriptor_set(mol)]

    X = np.array(X_CDS)

    CDS_names = ['OB$_{100}$', 'n_C', 'n_N', 'n_NO', 'n_COH', 'n_NOC', 'n_CO', 'n_H', 'n_F', 'n_N/n_C', 'n_CNO2',
    'n$_{\\ff{NNO}_2}$', 'n$_{\\ff{ONO}}$', 'n$_{\\ff{ONO}_2}$', 'n_$\\ff{CNN}$', 'n_$\\ff{NNN}$', 'n_CNO', 'n_CNH2', 'n_CN(O)C', 'n_CF', 'n_CNF']

    if (return_names):
        return CDS_names, X
    else:
        return X

def Estate_CDS_LBoB_featurizer(mol_list, predefined_bond_types=[], return_names=True):
    return Estate_CDS_SoB_featurizer(mol_list, predefined_bond_types=predefined_bond_types, return_names=return_names)
def Estate_CDS_SoB_featurizer(mol_list, predefined_bond_types=[], scaled=True, return_names=True):

    if (isinstance(mol_list, list) == False):
        mol_list = [mol_list]
    names_Estate, X_Estate = truncated_Estate_featurizer(mol_list, return_names=True )
    names_CDS, X_CDS = CDS_featurizer(mol_list, return_names=True)
    names_LBoB, X_LBoB = literal_bag_of_bonds(mol_list, predefined_bond_types=predefined_bond_types)

    X_combined = np.concatenate((X_Estate, X_CDS, X_LBoB), axis=1)

    if scaled:
        X_scaled = StandardScaler().fit_transform(X_combined)
    else:
        X_scaled = X_combined
        

    names_all = list(names_Estate)+list(names_CDS)+list(names_LBoB)
   
    if (return_names):
        return names_all, X_scaled
    else:
        return X_scaled
#finish

def modified_oxy_balance(mol):

    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
    n_atoms = mol.GetNumAtoms()

    OB = oxygen_balance_100(mol)

    correction = 100*(1.0*n_O2 + 1.8*n_O2 + 2.2*n_O3)/n_atoms

    return OB - correction


def return_atom_nums_modified_OB(mol):
    n_C = get_num_atom(mol, 6)
    n_N = get_num_atom(mol, 7)
    n_H = get_num_atom(mol, 1)
    n_F = get_num_atom(mol, 9)
    n_O1 = get_num_with_neighs(mol, 'O', {'N': 1})
    n_O2 = get_num_with_neighs(mol, 'O', {'N': 1,'C': 1})
    n_O3 = get_num_with_neighs(mol, 'O', {'C': 1})
    n_O4 = get_num_with_neighs(mol, 'O', {'C': 1,'H': 1})
    return [n_C, n_N, n_H, n_O1, n_O2, n_O3, n_O4, n_F]



def isRingAromatic(mol, bondRing):
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True
    
# function for ABOCH # OF CYCLES AND RING  
def Identify_Rings(mol):
    ri = mol.GetRingInfo()
    num_rings = len(ri.AtomRings())
    bondRing = ri.BondRings()
    num_aromatic_rings = 0
    for item in bondRing:
        if isRingAromatic(mol, item):
            num_aromatic_rings +=1
    
    # num_bezene = Chem.Fragments.fr_benzene(mol)
    return [num_rings, num_aromatic_rings]

# function for ABOCH counts of olefin and aromatic atom 
def Count_aromatic_olefin_atoms(mol):
    aromatic_carbon = Chem.MolFromSmarts("c")
    num_aroms = len(mol.GetSubstructMatches(aromatic_carbon))
    olefinic_carbon = Chem.MolFromSmarts("[C^2]")
    num_oles = len(mol.GetSubstructMatches(olefinic_carbon))
    return [num_aroms, num_oles]

# function of ABOCH # OF contiguous rotatable bonds 
from rdkit.Chem.Lipinski import RotatableBondSmarts
def find_bond_groups(mol):
    """Find groups of contiguous rotatable bonds and return them sorted by decreasing size"""
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = set([mol.GetBondBetweenAtoms(*ap).GetIdx() for ap in rot_atom_pairs])
    rot_bond_groups = []
    while (rot_bond_set):
        i = rot_bond_set.pop()
        connected_bond_set = set([i])
        stack = [i]
        while (stack):
            i = stack.pop()
            b = mol.GetBondWithIdx(i)
            bonds = []
            for a in (b.GetBeginAtom(), b.GetEndAtom()):
                bonds.extend([b.GetIdx() for b in a.GetBonds() if (
                    (b.GetIdx() in rot_bond_set) and (not (b.GetIdx() in connected_bond_set)))])
            connected_bond_set.update(bonds)
            stack.extend(bonds)
        rot_bond_set.difference_update(connected_bond_set)
        rot_bond_groups.append(tuple(connected_bond_set))
    return [len(tuple(sorted(rot_bond_groups, reverse = True, key = lambda x: len(x))))]

# function of ABOCH for # of conjugate double bond and aromatic bond
def Count_bonds(mol):
    #### Add H will increase the number of "realized" atoms
    
    num_bonds = len(mol.GetBonds())
    
    conj_double = 0
    arom_bond = 0
    ## Count the number of conjugate double bond
    for i in range(num_bonds):
        if (mol.GetBondWithIdx(i).GetBondType() == rdkit.Chem.rdchem.BondType.DOUBLE) and (mol.GetBondWithIdx(i).GetIsConjugated()):
            conj_double +=1
        if mol.GetBondWithIdx(i).GetIsAromatic():
            arom_bond+=1
    
    return [conj_double, arom_bond]

    
    

        
        