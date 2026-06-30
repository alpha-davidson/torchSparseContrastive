"""
name: O16_downstream_pipeline.py
 
Use this to create O16 data in different event sizes and trials to be used to evaluate benchmark vs. pretrained across different number of events.
 
date created: Jul 22, 2024
Last Edit: Apr 2026
Hakan Bora Yavuzkara
"""
 
import h5py
import numpy as np
import tqdm
import os
import pandas as pd
from sklearn import preprocessing
 

DATA_PATH = '/home/DAVIDSON/jugelina/benchmark_data/O16_data/'
ISOTOPE = 'O16'
SAMPLE_SIZE = 512

# Data size values (from training_pipeline.py calls)
# Each trials are the same for more accurate standard deviations
TRIALS = 10
DATA_SIZES = [30,45,60,90,120,150,180,210,240,270,300,360,420, 480, 570, 700, 830, 1000, 1200, 1400]

# 1 – convert raw h5 to numpy
def convert_data(data):
    """Convert raw h5 point-cloud to numpy arrays and save to DATA_PATH."""
    keys = list(data.keys())
 
    # event lengths
    event_lens = np.zeros(len(keys), int)
    for i in range(len(keys)):
        event_lens[i] = len(data[keys[i]])
    np.save(DATA_PATH + ISOTOPE + '_event_lens.npy', event_lens)
 
    # full event array  shape: (n_events, max_len, 6)
    # cols: x, y, z, time, amplitude, event_index
    event_data = np.zeros((len(keys), np.max(event_lens), 6), float)
    for n in tqdm.tqdm(range(len(keys)), desc='Converting h5'):
        name = keys[n]
        event = data[name]
        for i, e in enumerate(event):
            instant = np.array(list(e))
            event_data[n, i, :5] = instant[:5]
            event_data[n, i, 5]  = float(n)
 
    # fix the known empty event at (index 56437)
    event_data[56437, 0, 5] = 56437
    np.save(DATA_PATH + ISOTOPE + '_w_event_keys.npy', event_data)
    print(f'[convert_data] saved event array {event_data.shape}')
 

# 2 – attach track labels from CSV
def add_num_tracks():
    """
    Read O16_labels.csv (columns: 'Event number', 'Number of tracks') and
    build a dataset array of shape (n_labelled_events, max_len, 7).
 
    Layout of dataset returned:
        0-2 : x, y, z
        3   : amplitude (q)
        4   : event index
        5   : number of tracks (raw label)
        6   : event length

    Only the point at index 0 contains the event index, number of tracks, and event length information.
    """
    event_data = np.load(DATA_PATH + ISOTOPE + '_w_event_keys.npy')
    event_lens = np.load(DATA_PATH + ISOTOPE + '_event_lens.npy')
    df         = pd.read_csv(DATA_PATH + ISOTOPE + '_labels.csv', 
                             usecols=['Event number', 'Number of tracks'])
    
    labels  = np.array(df['Number of tracks'])
    indices = np.array(df['Event number'])
    max_len = int(np.max(event_lens))
 
    dataset = np.zeros((len(indices), max_len, 7), float)
    count   = 0
    for i in range(len(indices)):
        if not np.isnan(labels[i]):
            ev_num       = int(indices[i])
            ev_len       = int(event_lens[ev_num])
            dataset[count, :ev_len, :3] = event_data[ev_num, :ev_len, :3]  # x,y,z
            dataset[count, :ev_len,  3] = event_data[ev_num, :ev_len,  4]  # amplitude→q
            dataset[count, 0,        4] = event_data[ev_num, 0,        5]  # event index
            dataset[count, 0,        5] = labels[i]                        # track label
            dataset[count, 0,        6] = ev_len                           # event length
            count += 1
 
    dataset = dataset[:count]
    np.save(DATA_PATH + ISOTOPE + '_dataset.npy', dataset)
    print(f'[add_num_tracks] dataset shape: {dataset.shape}')
 
 
# 3 – simplify track-count labels  (from voxel pipeline simplify_class)
# This can certainly be updated to work differently
def simplify_class():
    """
    Re-map raw track counts to 3 classes:
        0,1,2 -> 0
        3     -> 1
        4,5   -> 2
    Stored back into dataset[:, 0, 5].
    """
    dataset = np.load(DATA_PATH + ISOTOPE + '_dataset.npy')
    labels  = dataset[:, 0, 5].astype(int)
    label_to_code        = np.array([0, 0, 0, 1, 2, 2])
    dataset[:, 0, 5]     = label_to_code[labels]
    np.save(DATA_PATH + ISOTOPE + '_dataset.npy', dataset)
    unique, counts = np.unique(dataset[:, 0, 5], return_counts=True)
    print(f'[simplify_class] class distribution: {dict(zip(unique.astype(int), counts))}')
 
# 4 – random up/down sample each event to exactly SAMPLE_SIZE points
def random_sample():
    """
    Each event is resampled to exactly SAMPLE_SIZE rows.
    Output shape: (n_events, SAMPLE_SIZE, 5)
        0-3: x, y, z, q
        4  : simplified track label  (scalar stored at [:, 0, 4])
    """
    
    dataset = np.load(DATA_PATH + ISOTOPE + '_dataset.npy')

    n_events = dataset.shape[0] 
    new_data = np.zeros((n_events, SAMPLE_SIZE, 5), float)

    for idx in tqdm.tqdm(range(n_events), desc='Random sampling'):
        points  = dataset[idx, :, :4]
        nonzero = points[points[:, 0] != 0]
        n       = nonzero.shape[0]

        if n >= SAMPLE_SIZE:
            chosen = np.random.choice(n, SAMPLE_SIZE, replace=False)
            new_data[idx, :, :4] = nonzero[chosen]
        else:
            new_data[idx, :n, :4] = nonzero
            need  = SAMPLE_SIZE - n
            extra = np.random.choice(n, need, replace=(need > n))
            new_data[idx, n:, :4] = nonzero[extra]

        new_data[idx, 0, 4] = dataset[idx, 0, 5]  # track label

    assert new_data.shape == (n_events, SAMPLE_SIZE, 5), 'Shape mismatch after sampling'
    np.save(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_sampled.npy', new_data)
    print(f'[random_sample] sampled array shape: {new_data.shape}')
 
 
# 5 – scale x,y,z,q features
def scale_data():
    """
    Standard-scale x, y, z to [0,1] and leave q unscaled.
    Label column [:, 0, 4] is left untouched.

    Scaling follows the analysis in plotting/exploreData.ipynb.
    """
    data = np.load(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_sampled.npy')
    
    for col in range(3):
        min_val = np.min(data[:, :, col])
        max_val = np.max(data[:, :, col])
        data[:, :, col] = (data[:, :, col] - min_val) / (max_val - min_val)
 
    assert np.sum(np.isnan(data)) == 0,  'NaNs after scaling'
    assert np.sum(np.isinf(data)) == 0,  'Infs after scaling'
 
    np.save(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_scaled.npy', data)
    print('[scale_data] scaling complete')
 
 
# 6 – 60 / 20 / 20 split. Constant val and test files  +  train pool which will be chopped.

#TODO, Get rid of split_later completely
def split_train_val_test(split_later=False):

    # Split_later is used to combine train and val files in order to keep test files constant, and train and val depend
    # on the number of data
    """
    Splits scaled data 60/20/20.
    Saves:
        O16_size512_train.npy   shape (0.6*N, 512, 5)
        O16_size512_val.npy     shape (0.2*N, 512, 5)
        O16_size512_test.npy    shape (0.2*N, 512, 5)
    And convenience feature/label files for val and test.
    """
    
    data         = np.load(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_scaled.npy')
    rand_shuffle = np.random.permutation(len(data))
    # If you want only the training data size to change
    if not split_later:
        print("7: 60/20/20 split")
        n          = len(data)
        test_end   = int(n * 0.2)
        val_end    = int(n * 0.4)
     
        test  = data[rand_shuffle[:test_end]]
        val   = data[rand_shuffle[test_end:val_end]]
        train = data[rand_shuffle[val_end:]]
     
        print(f'[split] train {train.shape}  val {val.shape}  test {test.shape}')
     
        base = DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE)
        np.save(base + '_train.npy', train)
        np.save(base + '_val.npy',   val)
        np.save(base + '_test.npy',  test)
     
        # constant feature / label files for val & test (used directly by training_pipeline)
        np.save(base + '_val_features.npy',  val[:, :, :4])
        np.save(base + '_val_labels.npy',    val[:, 0,  4])
        np.save(base + '_test_features.npy', test[:, :, :4])
        np.save(base + '_test_labels.npy',   test[:, 0,  4])
     
        assert np.sum(np.isnan(train)) == 0, 'NaNs in train'
        assert np.sum(np.isnan(val))   == 0, 'NaNs in val'
        assert np.sum(np.isnan(test))  == 0, 'NaNs in test'
     # If you want training and val data size to change, and test to remain constant.
    else:
        print("7: 80/20 split")
        n        = len(data)
        test_end = int(n * 0.2)

        test     = data[rand_shuffle[:test_end]]
        trainval = data[rand_shuffle[test_end:]]

        print(f'[split] trainval {trainval.shape}  test {test.shape}')

        base = DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE)
        np.save(base + '_trainval.npy',          trainval)
        np.save(base + '_test.npy',              test)

        # flat feature / label files
        np.save(base + '_trainval_features.npy', trainval[:, :, :4])
        np.save(base + '_trainval_labels.npy',   trainval[:, 0,  4])
        np.save(base + '_testlate_features.npy',     test[:, :, :4])
        np.save(base + '_testlate_labels.npy',       test[:, 0,  4])

        assert np.sum(np.isnan(trainval)) == 0, 'NaNs in trainval'
        assert np.sum(np.isnan(test))     == 0, 'NaNs in test'
        
 
 
# 7 – generate trial files for every data size
def generate_trials(split_later=False):
    """
    For each training size:
        - randomly draw N events from the training pool (without replacement)
        - save features [N, 512, 4] and labels [N] as separate .npy files
        - repeat for each 10 trials per data size
 
    Output folders (inside DATA_PATH):
        O16_size512_{N}train_features/trial_k.npy
        O16_size512_{N}train_labels/trial_k.npy
    """
    if split_later:
        train = np.load(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_trainval.npy')
    else:
        train = np.load(DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_train.npy')
    n_train = len(train)
    print(f'[generate_trials] training pool size: {n_train}')
 
    for N in DATA_SIZES:
        if N > n_train:
            print(f'  [SKIP] N={N} exceeds training pool ({n_train}), skipping.')
            continue
    
        feat_dir  = DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_' + str(N) + 'train_features/'
        label_dir = DATA_PATH + ISOTOPE + '_size' + str(SAMPLE_SIZE) + '_' + str(N) + 'train_labels/'
        os.makedirs(feat_dir,  exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
    
        for k in range(1, TRIALS + 1):
            chosen   = np.random.choice(n_train, N, replace=False)
            subset   = train[chosen]
            features = subset[:, :, :4]
            labels   = subset[:, 0,  4].astype(int)
    
            np.save(feat_dir  + f'trial_{k}.npy', features)
            np.save(label_dir + f'trial_{k}.npy', labels)
    
        print(f'  N={N:4d}  trials={TRIALS}  feat shape={features.shape}')
    
    print('[generate_trials] done.')

    
 

def main():
    os.makedirs(DATA_PATH, exist_ok=True)

    # one-time conversion (comment out after first run)
    print("1: convert h5 to npy")
    data_h5 = h5py.File(DATA_PATH + 'O16_run160.h5', 'r')
    convert_data(data_h5)
 
    print("2: attach track labels")
    add_num_tracks()
 
    print("3: simplify class label")
    simplify_class()

    print("4: random sample to 512 points")
    random_sample()

    print("5: scale features")
    scale_data()

    
    
    # run every time (or when you need fresh splits / trials)

    # If validation and training will be split later, set split_later = True for both following functions
    # Split_later must be true if you would prefer a changing training and validation set, and a constant test set
    # Send this as split_train_val_test(split_later=False) and generate_trials(split_later=False)
    
    #6
    split_train_val_test()
    
    print("7: generate trial files")
    generate_trials()
 
    print("Pipeline complete")
 
 
if __name__ == '__main__':
    main()