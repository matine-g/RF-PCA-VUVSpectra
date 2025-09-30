
import tensorflow as tf
import numpy as np
### Fix the seed to get the reproducible results
tf.random.set_seed(42)
np.random.seed(42)
from tensorflow.keras import Model, layers
from tensorflow.keras import Input
# from tensorflow_recommenders.layers.feature_interaction.dcn import Cross
from sklearn.metrics import accuracy_score


### Import tensorflow part
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import Model, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import rdkit


### Regression get results
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import statistics
from sklearn.model_selection import train_test_split
from Lib_feats_new import *


import sys
import os

from rdkit import Chem, DataStructs
import matplotlib.pyplot as plt

# Spektral is needed for SDF parsing helpers
try:
    from spektral.io import (
        _parse_header, _parse_counts_line, _parse_atoms_block, _parse_bonds_block,
        _parse_properties, _parse_data_fields, _get_atomic_num
    )
except Exception as e:
    raise ImportError(
        "Spektral is required. Install with: pip install 'spektral==1.*'"
    ) from e

#### Toolbox for Euclidean distance
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial import distance_matrix

import pandas as pd
import pickle


import warnings
warnings.filterwarnings("ignore")


class NeuralNet(Model):
    # Set layers.
    def __init__(self, n_hidden_1, n_hidden_2, num_out):
        super(NeuralNet, self).__init__()
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)
        # First fully-connected hidden layer.
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
        # Second fully-connecter hidden layer.
        self.out = layers.Dense(num_out)

    # Set forward pass.
    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
    
class Conv1D(Model):
    # Set layers.
    def __init__(self, n_filter1, n_filter2, num_out):
        super(Conv1D, self).__init__()
        # First Conv layer
        self.conv1 = layers.Conv1D(n_filter1, kernel_size=5, activation="relu")
        # First pooling layer
        self.maxpool1 = layers.MaxPool1D(pool_size=2)
        
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv1D(n_filter2, kernel_size=3, activation="relu")
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool1D(pool_size=2)
        
        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)
        
        self.out = layers.Dense(num_out)

    # Set forward pass.
    def call(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        
        return x
    
    
class LSTM(Model):
    # Set layers.
    def __init__(self, num_units, n_hidden, num_out):
        super(LSTM, self).__init__()
        # First Conv layer
        self.lstm_layer = layers.LSTM(units=num_units)
        self.fc = layers.Dense(n_hidden, activation=tf.nn.relu)
        self.out = layers.Dense(num_out)

    # Set forward pass.
    def call(self, x):
        
        x = self.lstm_layer(x)
        x = self.fc(x)
        x = self.out(x)
        
        return x

class Conv2D(Model):
    # Set layers.
    def __init__(self, n_filter1, n_filter2, num_out):
        super(Conv2D, self).__init__()
        # First Conv layer
        self.conv1 = layers.Conv2D(n_filter1, kernel_size=(2, 1), activation="relu")
        # First pooling layer
        self.maxpool1 = layers.MaxPool2D(pool_size=(2, 1))
        
        # Convolution Layer with 64 filters and a kernel size of 3.
        self.conv2 = layers.Conv2D(n_filter2, kernel_size=(2, 1), activation="relu")
        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. 
        self.maxpool2 = layers.MaxPool2D(pool_size=(2, 1))
        
        # Flatten the data to a 1-D vector for the fully connected layer.
        self.flatten = layers.Flatten()

        # Fully connected layer.
        self.fc1 = layers.Dense(1024)
        
        # Apply Dropout (if is_training is False, dropout is not applied).
        self.dropout = layers.Dropout(rate=0.5)
        
        self.out = layers.Dense(num_out)

    # Set forward pass.
    def call(self, x):
        
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        
        return x

def mse_loss(x, y):
    # Not much necessary but to make sure that the inputs are float numbers
    y = tf.cast(y, tf.float64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.keras.metrics.mean_squared_error(y,x)           
    # Average loss across the batch.
    return tf.reduce_mean(loss)

def get_results(y_true, y_pred):
    mse_all = []
    r2_all = []
    for i in range(len(y_true)):
            mse_all.append(mean_squared_error(y_true[i], y_pred[i]))
            r2_all.append(r2_score(y_true[i], y_pred[i]))
        
    result_mse = statistics.mean(mse_all) 
    result_r2 = statistics.mean(r2_all)
    
    
    
    return result_r2, result_mse

def get_results_new(y_true, y_pred):
    mse_all = []
    r2_all = []
    for i in range(len(y_true)):
            mse_all.append(mean_squared_error(y_true[i], y_pred[i]))
            r2_all.append(r2_score(y_true[i], y_pred[i]))
        
    result_mse = statistics.mean(mse_all) 
    result_r2 = statistics.mean(r2_all)
    
    
    
    return result_r2, result_mse

def get_results_update(y_true, y_pred):
    mse_all = []
    r2_all = []
    for i in range(len(y_true)):
            mse_all.append(mean_squared_error(y_true[i], y_pred[i]))
            r2_all.append(r2_score(y_true[i], y_pred[i]))

    return r2_all, mse_all


def run_optimization(x, y, model, optimizer):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = model(x)
        # Compute loss.
        loss = mse_loss(pred, y)
        
    # Variables to update, i.e. trainable variables.X
    trainable_variables = model.trainable_variables

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))
    

def run_process(train_x, test_x, train_y, test_y, model, batch_size, learning_rate, epochs, display_step):
    
    train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
    
    
   
    optimizer = tf.optimizers.Adam(learning_rate)
    for step, (batch_x, batch_y) in enumerate(train_data.take(epochs), 1):
        # Run the optimization to update W and b values.
        run_optimization(batch_x, batch_y, model, optimizer)
        
        if step % display_step == 0:
            pred = model(batch_x)
            loss = mse_loss(pred, batch_y)
            # acc = accuracy(pred, batch_y) #### There should be a change here
            pred_ = pred.numpy()
            batch_y_ = batch_y.numpy()
            r2_, mse_ = get_results(batch_y_, pred_)
            print("step: %i, loss: %f, R2 score: %f" % (step, loss, r2_))
            
    #### Final evaluation on the test set
    pred_test = model(test_x)
    # r2_test,_ = get_results(test_y, pred_test)
    # print("R2 of the method on the test set: ", r2_test)
    
    return pred_test

########## To check 06/07/22
def get_model(n_inputs, n_hidden, n_outputs):
    input_layer = Input(shape=(n_inputs,), name = 'input_layer')
    X1 = Dense(n_hidden, kernel_initializer='he_uniform', activation='relu')(input_layer)
    Y1 = Dense(n_outputs, name='spectra_output')(X1)
    model = Model(inputs= input_layer, outputs= [Y1])
    model.compile(loss={"spectra_output": 'mse'}, optimizer='adam')
    return model

#### The same option as Neural Net, let the built-in functions of tensorflow work themselves

def get_model2(n_inputs, n_hidden1, n_hidden2, loss, n_outputs):
    input_layer = Input(shape=(n_inputs,), name = 'input_layer')
    X1 = Dense(n_hidden1, kernel_initializer='he_uniform', activation='relu')(input_layer)
    X2 = Dense(n_hidden2, kernel_initializer='he_uniform', activation='relu')(X1)
    Y1 = Dense(n_outputs, name='spectra_output')(X2)
    model = Model(inputs= input_layer, outputs= [Y1])
    model.compile(loss={"spectra_output": loss}, optimizer='adam')
    return model


def predict_new_observation(file_path, pre_bond_types, models, wave_length, dataframe):
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    
    sppl = Chem.SDMolSupplier(file_path)
    for mol in sppl:
        if mol is not None:# some compounds cannot be loaded.z
                    smiles = Chem.MolToSmiles(mol)
                    
    mol_new = Chem.MolFromSmiles(smiles)
    mol_new = Chem.AddHs(mol_new)
    __,X_new_LBOB = literal_bag_of_bonds([mol_new], predefined_bond_types= pre_bond_types)
    X_new_CDS = np.array(return_combined_nums_update(mol_new)).reshape(1,-1)
    print(X_new_CDS.shape)
    
    ##### If you want to recheck: only the bond types may cause the errors
    # mol_train = list(data['Mols'].values)
    # mol_train.append(mol_new)
    # bond_types_, X_LBoB_ = literal_bag_of_bonds(aa)   #### If new molecule has new bond type, it will be added in the end
    
    
    #### Start the new prediction here
    X_new = np.concatenate((X_new_CDS, X_new_LBOB), axis=1)
    ##### Matches with the 
    preds = []
    for model in models:
            pred_test = model(X_new).numpy().tolist()[0]
            preds.append(pred_test)
    
    ######### Retrieve the y_true from the file
    positions = [i for i, letter in enumerate(file_path) if letter == '/']
    full_file_name = file_path[positions[-1]+1::]
    pos_space = full_file_name.find(' ')
    check_number = full_file_name[:pos_space]
    
    ### Trace back the spectra from given CAS number
    number_series = dataframe.iloc[0]
    if number_series[number_series == check_number].index.size > 0:
            col_name = number_series[number_series == check_number].index[0]
            # Retrieve the spectra 
            y_true = dataframe[[col_name]].values[1::].tolist()
            y_true = [item[0] for item in y_true]

    prediction = np.array(preds)
    final_pred = np.mean(prediction, axis=0)
    score = np.round(r2_score(y_true, final_pred),2)
    print(score)
    plt.plot(wave_length, y_true, 'b', label = "True")
    
    # plotting the line 2 points 
    plt.plot(wave_length, final_pred, 'r', label = "Predict")
    plt.xlabel('Wavelength')
    plt.text(250, 0.9, r'$R2 \ score =$'+str(score), fontdict=font)
    # Set the y axis label of the current axis.
    plt.ylabel('VUV spectra')
    # Set a title of the current axes.
    plt.title(full_file_name[:-4])
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    
    return score


def predict_new_observation_new(file_path, pre_bond_types, models, wave_length, dataframe):
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }
    
    sppl = Chem.SDMolSupplier(file_path)
    for mol in sppl:
        if mol is not None:# some compounds cannot be loaded.z
                    smiles = Chem.MolToSmiles(mol)
                    
    mol_new = Chem.MolFromSmiles(smiles)
    mol_new = Chem.AddHs(mol_new)
    __,X_new_LBOB = literal_bag_of_bonds([mol_new], predefined_bond_types= pre_bond_types)
    X_new_CDS = np.array(return_combined_nums_update(mol_new)).reshape(1,-1)
    X_new_rings = np.array(Identify_Rings(mol_new)).reshape(1,-1)
    X_new_count = np.array(Count_aromatic_olefin_atoms(mol_new)).reshape(1,-1)
    X_new_bond_group = np.array(find_bond_groups(mol_new)).reshape(1,-1)

    
    #### Start the new prediction here
    X_new = np.concatenate((X_new_CDS, X_new_LBOB, X_new_rings, X_new_count, X_new_bond_group), axis=1)
    ##### Matches with the 
    preds = []
    for model in models:
            pred_test = model(X_new).numpy().tolist()[0]
            preds.append(pred_test)
    
    ######### Retrieve the y_true from the file
    positions = [i for i, letter in enumerate(file_path) if letter == '/']
    full_file_name = file_path[positions[-1]+1::]
    pos_space = full_file_name.find(' ')
    check_number = full_file_name[:pos_space]
    
    ### Trace back the spectra from given CAS number
    number_series = dataframe.iloc[0]
    if number_series[number_series == check_number].index.size > 0:
            col_name = number_series[number_series == check_number].index[0]
            # Retrieve the spectra 
            y_true = dataframe[[col_name]].values[1::].tolist()
            y_true = [item[0] for item in y_true]

    prediction = np.array(preds)
    final_pred = np.mean(prediction, axis=0)
    score = np.round(r2_score(y_true, final_pred),2)
    print(score)
    plt.plot(wave_length, y_true, 'b', label = "True")
    
    # plotting the line 2 points 
    plt.plot(wave_length, final_pred, 'r', label = "Predict")
    plt.xlabel('Wavelength')
    plt.text(250, 0.9, r'$R2 \ score =$'+str(score), fontdict=font)
    # Set the y axis label of the current axis.
    plt.ylabel('VUV spectra')
    # Set a title of the current axes.
    plt.title(full_file_name[:-4])
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.show()
    
    return score


def isRingAromatic(mol, bondRing):
        for id in bondRing:
            if not mol.GetBondWithIdx(id).GetIsAromatic():
                return False
        return True
    
    
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

def Count_aromatic_olefin_atoms(mol):
    aromatic_carbon = Chem.MolFromSmarts("c")
    num_aroms = len(mol.GetSubstructMatches(aromatic_carbon))
    olefinic_carbon = Chem.MolFromSmarts("[C^2]")
    num_oles = len(mol.GetSubstructMatches(olefinic_carbon))
    return [num_aroms, num_oles]

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

###### 3D feats: Linh proposed on May 19th
def distance_type_1(atom, coords_atom, atoms_new):   ### Function for all the heavy atoms
    number_test = get_atomic_num(atom)
    aggs = ['mean', 'max', 'min']
    features_name = ['distance_'+item +"_" + atom for item in aggs]
    
    indices = [idx for idx, element in enumerate(atoms_new) if element == number_test]
    
    if len(indices)>0:   ## If the molecule has that heavy atom
        coords_= coords_atom[indices]
        indices_non_ = [idx for idx, element in enumerate(atoms_new) if element != number_test]
        coords_non_ = coords_atom[indices_non_]
        
        distances_pairwise_ = euclidean_distances(coords_non_, coords_)
        X = [np.mean(distances_pairwise_), np.max(distances_pairwise_), np.min(distances_pairwise_)]
    else:           ## If the molecule does not have that heavy atom
        X = [0,0,0]
        
    return features_name, X

def distance_type_2(coords_atom, atoms_new):  ### For now, only for 'C'
    
    indices_C = [idx for idx, element in enumerate(atoms_new) if element == 6]
    aggs = ['mean', 'max', 'min']
    features_name = ['distance_pairwise_'+item +"_C" for item in aggs]
    
    if len(indices_C) > 1:
        coords_C = coords_atom[indices_C]
        distances_C = euclidean_distances(coords_C, coords_C)
        aa = np.triu(np.round(distances_C,2))
        bb = aa[np.nonzero(aa)]   
        X = [np.mean(bb), np.max(bb), np.min(bb)]
    else:
        X = [0,0,0]
        
    return features_name, X

def  Extract_3D_feats(loc, df, heavy_atoms):
    filename_sdf  = df['File Path'][loc]


    with open(filename_sdf) as f:
        data_new = f.read().split('$$$$\n')
        
    sdf_out = {}            # Empty dictionary to save the data
    sdf = data_new[0].split('\n')
    sdf_out['name'], sdf_out['details'], sdf_out['comment'] = _parse_header(sdf)
    sdf_out['n_atoms'], sdf_out['n_bonds'] = _parse_counts_line(sdf)
    sdf_out['atoms'] = _parse_atoms_block(sdf, sdf_out['n_atoms'])
    sdf_out['bonds'] = _parse_bonds_block(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['properties'] = _parse_properties(sdf, sdf_out['n_atoms'], sdf_out['n_bonds'])
    sdf_out['data'] = _parse_data_fields(sdf)
        
    
    atoms_new = []
    for item in sdf_out['atoms']:
        atoms_new.append(item['atomic_num'])
        
    atom_coords_new = []
    for item in sdf_out['atoms']:
        xyz_new = [float(w) for w in item['coords']]
        atom_coords_new.append(xyz_new)
        
        
    coords = np.array(atom_coords_new)
    features_name_type1 = []
    X_3D_type1 = []
    for item in heavy_atoms:
        features_name, X = distance_type_1(item, coords, atoms_new)
        features_name_type1 += features_name
        X_3D_type1 += X  
    
    features_C, X_C = distance_type_2(coords, atoms_new)
    
    features_3D_all = features_name_type1 + features_C
    X_3D_all = X_3D_type1 + X_C
    
    df_= pd.DataFrame(columns= features_3D_all, data=np.expand_dims(np.array(X_3D_all), axis = 0))
    
    return df_

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
