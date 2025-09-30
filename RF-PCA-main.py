import time
import numpy as np
import pandas as pd
from Lib_feats_new import *
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn import tree as sktree
import os
from sklearn.metrics import mean_squared_error, r2_score
from rdkit import Chem

start_time = time.time()

### Load the data
df = pd.read_excel('C:/Users/QS/Downloads/linh/VUV_big_regression_updated/datasets/big_spectra_normalized.xlsx')

names_small = []
for file_path in df.Path:
    positions = [i for i, letter in enumerate(file_path) if letter == '/']
    full_file_name = file_path[positions[-1]+1::]
    pos_space = full_file_name.find(' ')
    check_number = full_file_name[:pos_space]   # CAS number
    name = full_file_name[pos_space+1:-4]
    names_small.append(name)

names_to_exclude = ['1-Bromo-3-nitrobenzene', 'Fluoromethane', '4-Allylphenol',
                    'p-Anisaldehyde', 'Carbon monoxide', 'Methyl iodide',
                    '1-Bromo-1-propene', 'Nitrous oxide']

to_exclude = [i for i in range(len(names_small)) if names_small[i] in names_to_exclude]
df.drop(df.index[to_exclude], inplace=True)
df.reset_index(inplace=True)

columns = ['index']
df.drop(columns, inplace=True, axis=1)

### Retrieve the spectra
all_spectra = []
for k in range(len(df)):
    spectra = df.loc[k, 'Spectra']
    i = list(spectra.split(", "))
    i[0] = i[0][1::]
    i[-1] = i[-1][:-1]
    sanu_float = [float(j) for j in i]
    all_spectra.append(sanu_float)

def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def get_results_update(y_true, y_pred):

    mse_all, r2_all = [], []
    for i in range(len(y_true)):
        mse_all.append(mean_squared_error(y_true[i], y_pred[i]))
        r2_all.append(r2_score(y_true[i], y_pred[i]))
    return r2_all, mse_all

all_spectra_refined = []
for spectra in all_spectra:
    denoise = smooth(spectra, 3)
    all_spectra_refined.append(denoise)

y = np.array(all_spectra_refined)


#######################################
data = df.copy()
data['Mols'] = data['SMILES'].apply(Chem.MolFromSmiles)
data['Mols'] = data['Mols'].apply(Chem.AddHs)

fingerprint_df = pd.read_csv('C:/Users/QS/Downloads/orgfingerprints.csv')
all_fingerprints_array = fingerprint_df.drop(columns=['Molecule_Name']).values
molecule_names_array = fingerprint_df['Molecule_Name'].values
fingerprint_dict = dict(zip(molecule_names_array, all_fingerprints_array))

matched_fingerprints = []
not_found_count = 0
for name in names_small:
    name_processed = name.replace(" ", "").lower()
    found = False
    for molecule_name in molecule_names_array:
        molecule_name_processed = molecule_name.replace(" ", "").lower()
        if name_processed in molecule_name_processed:
            matched_fingerprints.append(fingerprint_dict[molecule_name])
            found = True
            break
    if not found:
        print(f"Warning: No fingerprint found for molecule {name}")
        matched_fingerprints.append(np.zeros(all_fingerprints_array.shape[1]))
        not_found_count += 1

X_Estate = np.array(matched_fingerprints)
print(f"Number of molecules not found: {not_found_count}")

bond_types, X_LBoB = literal_bag_of_bonds(list(data['Mols']))

data['Oxygen Balance_100'] = data['Mols'].apply(oxygen_balance_100)
data['Oxygen Balance_1600'] = data['Mols'].apply(oxygen_balance_1600)
data['modified OB'] = data['Mols'].apply(modified_oxy_balance)
data['OB atom counts'] = data['Mols'].apply(return_atom_nums_modified_OB)
data['combined_nums'] = data['Mols'].apply(return_combined_nums_update)

X_OB100 = np.array(list(data['Oxygen Balance_100'])).reshape(-1, 1)
X_OB1600 = np.array(list(data['Oxygen Balance_1600'])).reshape(-1, 1)
X_OBmod = np.array(list(data['modified OB'])).reshape(-1, 1)
X_combined = np.array(list(data['combined_nums']))
X_Estate_combined = np.concatenate((X_Estate, X_combined), axis=1)
X_Estate_combined_lit_BoB = Estate_CDS_LBoB_featurizer(list(data['Mols']))
X_CustDesrip_lit_BoB = np.concatenate((X_combined, X_LBoB), axis=1)

data['Rings'] = data['Mols'].apply(Identify_Rings)
X_rings = np.array(list(data['Rings']))

data['Count'] = data['Mols'].apply(Count_aromatic_olefin_atoms)
X_count = np.array(list(data['Count']))

data['Bond_Group'] = data['Mols'].apply(find_bond_groups)
X_bond_group = np.array(list(data['Bond_Group']))

data['Bond_types'] = data['Mols'].apply(Count_bonds)
X_bond_types = np.array(list(data['Bond_types']))

X_new = np.concatenate((X_rings, X_count, X_bond_group, X_bond_types), axis=1)
X_new1 = np.concatenate((X_Estate_combined, X_LBoB, X_rings, X_count, X_bond_group, X_bond_types), axis=1)

featurization_dict = {
    "C.D.S+LBoB": X_CustDesrip_lit_BoB,
    'Estate+LBOB': np.concatenate((X_Estate, X_LBoB), axis=1),
    "Estate+CDS+LBoB": X_Estate_combined_lit_BoB[1],
    "BoB+OB": np.concatenate((X_LBoB, X_OB100, X_OB1600), axis=1),
    "Estate+CDS+LBoB+OB": np.concatenate((X_Estate_combined_lit_BoB[1], X_OB100, X_OB1600), axis=1),
    "New feat alone": X_new,
    "Estate": X_Estate,
    "CDS": X_combined,
    'New feat 1': X_new1
}

max_depth = 30
n_estimators = 100
estimators = {
    "Random Forest": [n_estimators, max_depth]
    
}

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=0)

def apply_pca_to_y(y):
    pca = PCA(n_components=0.99)
    y_pca = pca.fit_transform(y)
    print(f"Original shape (y): {y.shape}, Transformed shape (y_pca): {y_pca.shape}")
    return pca, y_pca

pca_y, y_pca = apply_pca_to_y(y)

r2_test_all = {}
for estimator, params in estimators.items():
    print(estimator)
    r2_test = {}
    for feature_name, feature in featurization_dict.items():
        feature_start_time = time.time()
        X = featurization_dict[feature_name]
        r2_test[feature_name] = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y_pca[train_index], y_pca[test_index]
            if estimator == "Random Forest":
                regr_rf = RandomForestRegressor(n_estimators=params[0], max_depth=params[1], random_state=42)
                regr_rf.fit(X_train, y_train)
                y_rf = regr_rf.predict(X_test)
                y_rf_original = pca_y.inverse_transform(y_rf)
                y_test_original = pca_y.inverse_transform(y_test)
                r2_test_RF, _ = get_results_update(y_test_original, y_rf_original)
                r2_test[feature_name].append(r2_test_RF)
                del regr_rf
        feature_time = time.time() - feature_start_time
        r2_mean = np.mean([item for sublist in r2_test[feature_name] for item in (sublist if isinstance(sublist, list) else [sublist])])
        print(f"Total execution time for {feature_name} in {estimator}: {feature_time:.2f} seconds")
        print(f"{feature_name} {r2_mean}")
    r2_test_all[estimator] = r2_test

Estate_names = ['-CH3', '=CH2', '—CH2—', '\\#CH', '=CH-', 'aCHa', '>CH-', '=c=', '\\#C-', '=C$<$', 'aCa',
    'aaCa', '$>$C$<$', '-NH3[+1]', '-NH2', '-NH2-[+1]', '=NH', '-NH-', 'aNHa', '\\#N', '$>$NH-[+1]',
    '=N—', 'aNa', '$>$N—', '—N$<$$<$', 'aaNs', '$>$N$<$[+1]', '-OH', '=0', '-0-', 'aOa']

X_new1_names = Estate_names + [
    'OB 100', 'number of C', 'number of N', 'number of O1', 'number of O2', 'number of O3', 'number of O4',
    'number of H', 'number of Fluorine', 'number of chlorine', 'number of Bromine', 'NC ratio',
    'CNO2', 'NNO2', 'ONO', 'ONO2', 'CNN', 'NNN', 'CN0', 'CNH2', 'CN(O)C', 'CF', 'CNF',
    'number of C-N', 'number of C:C', 'number of N=O', 'number of N-O', 'number of H-N',
    'number of C-H', 'number of C-O', 'number of H-O', 'number of C=C', 'number of C-C',
    'number of Br-C', 'number of C/C', 'number of C=O', 'number of C-F', 'number of C:O',
    'number of C-Cl', 'number of C#N', 'number of C:S', 'number of N:N', 'number of C:N',
    'number of C-S', 'number of C-Si', 'number of O-Si', 'number of C\\C', 'number of C/O',
    'number of C=N', 'number of C=S', 'number of C#C', 'number of N-N', 'number of H-S',
    'number of P-S', 'number of P=S', 'number of O-P', 'number of O=S', 'number of C-I',
    'number of S-S', 'number of O=O', 'number of C/N', 'number of N=N', 'number of O-S',
    'number of N-S', 'number of O=P', 'number of C/Cl', 'number of C-P', 'number of C\\Cl',
    'number of rings', 'number of aromatic rings', 'number of aromatic atoms',
    'number of olefin atoms', 'number of contigous rotatble bond groups',
    'conjugated double bonds', 'number of aromatic bonds'
]

assert len(X_new1_names) == X_new1.shape[1]

# Category-colored plot WITHOUT abbreviations (TOP features only) 
ESTATE_SET = set(Estate_names)
OB_SET = {'OB 100'}
CDS_SET = {'number of C', 'number of N', 'number of O1', 'number of O2', 'number of O3', 'number of O4',
           'number of H', 'number of Fluorine', 'number of chlorine', 'number of Bromine', 'NC ratio',
           'CNO2', 'NNO2', 'ONO', 'ONO2', 'CNN', 'NNN', 'CN0', 'CNH2', 'CN(O)C', 'CF', 'CNF'}
COCB_SET = {'number of C-N', 'number of C:C', 'number of N=O', 'number of N-O', 'number of H-N',
            'number of C-H', 'number of C-O', 'number of H-O', 'number of C=C', 'number of C-C',
            'number of Br-C', 'number of C/C', 'number of C=O', 'number of C-F', 'number of C:O',
            'number of C-Cl', 'number of C#N', 'number of C:S', 'number of N:N', 'number of C:N',
            'number of C-S', 'number of C-Si', 'number of O-Si', 'number of C\\C', 'number of C/O',
            'number of C=N', 'number of C=S', 'number of C#C', 'number of N-N', 'number of H-S',
            'number of P-S', 'number of P=S', 'number of O-P', 'number of O=S', 'number of C-I',
            'number of S-S', 'number of O=O', 'number of C/N', 'number of N=N', 'number of O-S',
            'number of N-S', 'number of O=P', 'number of C/Cl', 'number of C-P', 'number of C\\Cl'}
ABOCH_SET = {'number of rings', 'number of aromatic rings', 'number of aromatic atoms',
             'number of olefin atoms', 'number of contigous rotatble bond groups',
             'conjugated double bonds', 'number of aromatic bonds'}

CAT_COLORS = {
    'Estate': '#A3C9E2',
    'CDS':    '#B5E5CF',
    'COCB':   '#F9D6A5',
    'OB':     '#D3B6E0',
    'ABOCH':  '#F5B7B1',
    'Unknown':'#D5DBDB'
}

# Estate features to mark with an asterisk if they show up in the Top-N
ASTERISK_ESTATE = {
    '-CH3', '=CH2', '—CH2—', '\\#CH', '=CH-', 'aCHa', '>CH-',
    '-NH3[+1]', '-NH2', '-NH2-[+1]', '=NH', '-NH-', 'aNHa',
    '$>$N$<$[+1]', '-OH'
}

def feature_category(name: str) -> str:
    if name in ESTATE_SET: return 'Estate'
    if name in CDS_SET: return 'CDS'
    if name in COCB_SET: return 'COCB'
    if name in OB_SET: return 'OB'
    if name in ABOCH_SET: return 'ABOCH'
    return 'Unknown'

def plot_top_features_with_categories(X, y, feature_names, top_n=18,
                                      save_path="top_featureimportance.tiff", dpi=300):
    rf = RandomForestRegressor(n_estimators=100, max_depth=30, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    dfp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    dfp['Category'] = dfp['Feature'].apply(feature_category)

    # Take Top-N AFTER computing categories
    dfp = dfp.sort_values(by='Importance', ascending=False).head(top_n).copy()

    # Add asterisk for selected Estate features that appear in Top-N
    def star_label(row):
        base = row['Feature']
        if row['Category'] == 'Estate' and base in ASTERISK_ESTATE:
            base = '* ' + base
        return f"{base} ({row['Category']})"

    dfp['Labeled'] = dfp.apply(star_label, axis=1)

    # Colors by category 
    colors = dfp['Category'].map(CAT_COLORS)

    plt.figure(figsize=(12, 5))
    plt.barh(dfp['Labeled'], dfp['Importance'], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel('Feature Importance')

    # Legend stays the same
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in dfp['Category'].unique()]
    plt.legend(handles=handles, loc='lower right', title='Descriptor Sets')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

# Call the function for X_new1 
plot_top_features_with_categories(X_new1, y_pca, X_new1_names, top_n=18)

