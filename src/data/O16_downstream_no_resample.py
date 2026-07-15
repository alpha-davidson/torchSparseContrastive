"""
name: O16_downstream(no_resample).py

Using the O16 dataset for downstream tasks without resampling. To evaluate 
the model's performance on this dataset, we will use the original distribution 
of samples without any resampling techniques.

Date created: Jul 13, 2026
Last Edited: Jul 2026
Tony Mallen-Ntiador 
"""

import tqdm
import numpy as np
import h5py
import os
import pandas as pd
from sklearn import preprocessing

# from src.data.O16_downstream_pipeline import generate_trials, split_train_val_test

DATA_PATH = '/home/DAVIDSON/tomallenntiador/torchSparseContrastive/data/'
ISOTOPE = 'O16'
OUT_PREFIX = ISOTOPE + '_UNSAMPLED'

# Keep this False during normal use. The raw conversion creates the large
# O16_w_event_keys.npy file and should only be rerun intentionally.
RUN_CONVERT = False

TRIALS = 10
# number of events in downstream tasks
DATA_SIZES = [30,45,60,90,120,150,180,210,240,270,300,360,420, 480, 570, 700, 830, 1000, 1200, 1400]

# 1 - make sure the data/label files exist
def ensure_labels_csv_exists():
    """
    Ensure O16_labels.csv exists.

    If missing, create a template CSV with one row per event:
        Event number, Number of tracks

    The Number of tracks column is left blank because true labels
    cannot be inferred from the h5 file.
    """
    labels_path = DATA_PATH + ISOTOPE + '_labels.csv'
    event_lens_path = DATA_PATH + ISOTOPE + '_event_lens.npy'

    if os.path.exists(labels_path):
        return labels_path

    if not os.path.exists(event_lens_path):
        raise FileNotFoundError(
            f"Missing both labels CSV and event lengths file:\n"
            f"  labels: {labels_path}\n"
            f"  event_lens: {event_lens_path}\n"
            f"Run convert_data() first so the number of events is known."
        )

    event_lens = np.load(event_lens_path)
    n_events = len(event_lens)

    df = pd.DataFrame({
        'Event number': np.arange(n_events, dtype=int),
        'Number of tracks': np.nan,
    })

    df.to_csv(labels_path, index=False)

    raise FileNotFoundError(
        f"Labels file was missing, so I created a blank template at:\n"
        f"  {labels_path}\n\n"
        f"It has {n_events} rows, one per event.\n"
        f"Fill in the 'Number of tracks' column, then rerun the pipeline."
    )

# 2 - convert the h5py file to numpy arrays
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

# removed add_num_tracks()
# sike, uno_reverse.
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
    labels_path = ensure_labels_csv_exists()

    df = pd.read_csv(
        labels_path,
        usecols=['Event number', 'Number of tracks']
    )
    
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
    if labels.min() < 0 or labels.max() >= len(label_to_code):
        raise ValueError(
            f"Unexpected raw track label range: min={labels.min()}, max={labels.max()}. "
            f"Expected labels in [0, {len(label_to_code) - 1}]."
        )
    dataset[:, 0, 5]     = label_to_code[labels]
    np.save(DATA_PATH + ISOTOPE + '_dataset.npy', dataset)
    unique, counts = np.unique(dataset[:, 0, 5], return_counts=True)
    print(f'[simplify_class] class distribution: {dict(zip(unique.astype(int), counts))}')

# changed from original to use a variable number of points in training instead
# of the fixed resample to SAMPLE_SIZE=512 points
def reformat_columns():
    """
    Remap the 7-col dataset (x,y,z,q,event_idx,label,event_len)
    into a 5-col array (x,y,z,q,label) without resampling.

    Output shape: (n_events, max_len, 5)
        0-3: x, y, z, q
        4  : simplified track label (scalar at [:, 0, 4])
    Also saves event lengths for downstream padding masks.
    """
    dataset = np.load(DATA_PATH + ISOTOPE + '_dataset.npy')
    n_events, max_len, _ = dataset.shape
    event_lens = dataset[:, 0, 6].astype(int)

    new_data = np.zeros((n_events, max_len, 5), float)
    for idx in range(n_events):
        n = event_lens[idx]
        new_data[idx, :n, :4] = dataset[idx, :n, :4]  # x, y, z, q
        new_data[idx, 0,  4]  = dataset[idx, 0,  5]   # class label

    np.save(DATA_PATH + OUT_PREFIX + '_all.npy', new_data)
    np.save(DATA_PATH + OUT_PREFIX + '_event_lens.npy', event_lens)
    print(f'[reformat_columns] output shape: {new_data.shape}')


# 4 - data scaling to x,y,z,q features
def scale_data():
    """
    Standard-scale x, y, z to [0,1] and leave q unscaled.
    Label column [:, 0, 4] is left untouched.

    Scaling follows the analysis in plotting/exploreData.ipynb.
    """
    data = np.load(DATA_PATH + OUT_PREFIX + '_all.npy')
    event_lens = np.load(DATA_PATH + OUT_PREFIX + '_event_lens.npy')
    
    for col in range(3):
        # global min_max from real points only
        all_real = np.concatenate(
            [data[i, :event_lens[i], col] for i in range(len(data))]
        )
        min_val, max_val = all_real.min(), all_real.max()

        # scale only real rows per event, padding stays zero
        for i in range(len(data)):
            n = event_lens[i]
            data[i, :n, col] = (data[i, :n, col] - min_val) / (max_val - min_val)
 
    assert np.sum(np.isnan(data)) == 0,  'NaNs after scaling'
    assert np.sum(np.isinf(data)) == 0,  'Infs after scaling'
 
    np.save(DATA_PATH + OUT_PREFIX + '_scaled.npy', data)
    print('[scale_data] scaling complete')
    
#TODO, Get rid of split_later completely, inherited from Hakan file
# will experiment and later implement in the future.
def split_train_val_test(split_later=False):
    # Split_later is used to combine train and val files in order to keep test files constant, and train and val depend
    # on the number of data
    """
    Splits scaled data 60/20/20.
    Saves:
        O16_UNSAMPLED_train.npy   shape (0.6*N, max_len, 5)
        O16_UNSAMPLED_val.npy     shape (0.2*N, max_len, 5)
        O16_UNSAMPLED_test.npy    shape (0.2*N, max_len, 5)
        O16_UNSAMPLED_*_lens.npy  true unpadded lengths for each split
    And convenience feature/label files for val and test.
    """
    data = np.load(DATA_PATH + OUT_PREFIX + '_scaled.npy')
    event_lens = np.load(DATA_PATH + OUT_PREFIX + '_event_lens.npy')
    rand_shuffle = np.random.permutation(len(data))
    # If you want only the training data size to change
    if not split_later:
        print("7: 60/20/20 split")
        n          = len(data)
        test_end   = int(n * 0.2)
        val_end    = int(n * 0.4)
     
        test_idx  = rand_shuffle[:test_end]
        val_idx   = rand_shuffle[test_end:val_end]
        train_idx = rand_shuffle[val_end:]

        test  = data[test_idx]
        val   = data[val_idx]
        train = data[train_idx]

        test_lens  = event_lens[test_idx]
        val_lens   = event_lens[val_idx]
        train_lens = event_lens[train_idx]
     
        print(f'[split] train {train.shape}  val {val.shape}  test {test.shape}')
     
        base = DATA_PATH + OUT_PREFIX
        np.save(base + '_train.npy', train)
        np.save(base + '_val.npy',   val)
        np.save(base + '_test.npy',  test)
        np.save(base + '_train_lens.npy', train_lens)
        np.save(base + '_val_lens.npy',   val_lens)
        np.save(base + '_test_lens.npy',  test_lens)
     
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

        test_idx     = rand_shuffle[:test_end]
        trainval_idx = rand_shuffle[test_end:]

        test     = data[test_idx]
        trainval = data[trainval_idx]

        test_lens     = event_lens[test_idx]
        trainval_lens = event_lens[trainval_idx]

        print(f'[split] trainval {trainval.shape}  test {test.shape}')

        base = DATA_PATH + OUT_PREFIX
        np.save(base + '_trainval.npy',          trainval)
        np.save(base + '_test.npy',              test)
        np.save(base + '_trainval_lens.npy',     trainval_lens)
        np.save(base + '_test_lens.npy',         test_lens)

        # flat feature / label files
        np.save(base + '_trainval_features.npy', trainval[:, :, :4])
        np.save(base + '_trainval_labels.npy',   trainval[:, 0,  4])
        np.save(base + '_testlate_features.npy',     test[:, :, :4])
        np.save(base + '_testlate_labels.npy',       test[:, 0,  4])

        assert np.sum(np.isnan(trainval)) == 0, 'NaNs in trainval'
        assert np.sum(np.isnan(test))     == 0, 'NaNs in test'
        

def generate_trials(split_later=False):
    """
    For each training size:
        - randomly draw N events from the training pool (without replacement)
        - save features [N, max_len, 4], labels [N], and lengths [N] as separate .npy files
        - repeat for each 10 trials per data size
 
    Output folders (inside DATA_PATH):
        O16_UNSAMPLED_train_features/trial_k.npy
        O16_UNSAMPLED_train_labels/trial_k.npy
    """
    if split_later:
        train = np.load(DATA_PATH + OUT_PREFIX + '_trainval.npy')
        train_lens = np.load(DATA_PATH + OUT_PREFIX + '_trainval_lens.npy')
    else:
        train = np.load(DATA_PATH + OUT_PREFIX + '_train.npy')
        train_lens = np.load(DATA_PATH + OUT_PREFIX + '_train_lens.npy')
    n_train = len(train)
    print(f'[generate_trials] training pool size: {n_train}')

    for N in DATA_SIZES:
        if N > n_train:
            print(f'  [SKIP] N={N} exceeds training pool ({n_train}), skipping.')
            continue

        feat_dir  = DATA_PATH + OUT_PREFIX + '_' + str(N) + 'train_features/'
        label_dir = DATA_PATH + OUT_PREFIX + '_' + str(N) + 'train_labels/'
        lens_dir  = DATA_PATH + OUT_PREFIX + '_' + str(N) + 'train_lens/'
        os.makedirs(feat_dir,  exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(lens_dir,  exist_ok=True)

        for k in range(1, TRIALS + 1):
            chosen   = np.random.choice(n_train, N, replace=False)
            subset   = train[chosen]
            features = subset[:, :, :4]
            labels   = subset[:, 0,  4].astype(int)
            lengths  = train_lens[chosen]
    
            np.save(feat_dir  + f'trial_{k}.npy', features)
            np.save(label_dir + f'trial_{k}.npy', labels)
            np.save(lens_dir  + f'trial_{k}.npy', lengths)

        print(f'  N={N:4d}  trials={TRIALS}  feat shape={features.shape}')
    
    print('[generate_trials] done.')

def main():
    os.makedirs(DATA_PATH, exist_ok=True)

    # One-time conversion. This is skipped by default to avoid accidentally
    # overwriting/rebuilding the large raw O16_w_event_keys.npy file.
    raw_path = DATA_PATH + ISOTOPE + '_w_event_keys.npy'
    lens_path = DATA_PATH + ISOTOPE + '_event_lens.npy'
    if RUN_CONVERT or not (os.path.exists(raw_path) and os.path.exists(lens_path)):
        print("1: convert h5 to npy")
        with h5py.File(DATA_PATH + 'O16_run160.h5', 'r') as data_h5:
            convert_data(data_h5)
    else:
        print("1: raw converted files already exist; skipping h5 conversion")

    print("2: attach track labels")
    add_num_tracks()

    print("3: simplify class label")
    simplify_class()

    print("4: columns reformated to a 5 column array")
    reformat_columns()

    print("5: scale features")
    scale_data()

    
    
    # run every time (or when you need fresh splits / trials)

    # If validation and training will be split later, set split_later = True for both following functions
    # Split_later must be true if you would prefer a changing training and validation set, and a constant test set
    # Send this as split_train_val_test(split_later=False) and generate_trials(split_later=False)
    
    # number 6
    split_train_val_test()
    
    print("7: generate trial files")
    generate_trials()
 
    print("Pipeline complete")
 

if __name__ == '__main__':
    main()  
