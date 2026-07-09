"""
convert-data.py
---------------
One-time conversion of the raw O16 AT-TPC HDF5 file to numpy arrays.

Input
-----
    data/O16_run160.h5  — HDF5 file; each key is an event name, each value
                          is a structured array of hits with fields:
                          x, y, z, time_bucket, amplitude

Output (written to data/)
------
    O16_event_lens.npy        — (n_events,) int array of hit counts per event
    O16_w_event_keys.npy      — (n_events, max_hits, 6) float array
                                columns: x, y, z, time_bucket, amplitude, event_index

Usage
-----
    python convert-data.py

NOTE: O16_downstream_pipeline.py contains the same logic as part of its
full pipeline. This script is a standalone one-shot helper kept for reference.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import tqdm

data = h5py.File('data/O16_run160.h5','r')
keys = list(data.keys())

#making array of event lengths
event_lens = np.zeros(len(keys), int)
for i in range(len(keys)):
    event = keys[i]
    event_lens[i] = len(data[event])
    
np.save('data/O16_event_lens', event_lens)

#making a numpy array of data
event_data = np.zeros((len(keys), np.max(event_lens), 6), float)
for n in tqdm.tqdm(range(len(keys))):
    name = keys[n]
    event = data[name]
    ev_len = len(event)
    for i,e in enumerate(event):
        instant = np.array(list(e))
        event_data[n,i,:5] = instant[:5]
        event_data[n,i,5] = float(n) #storing the event index
        
event_data[56437,0,5] = 56437 #fixing the empty event ('Event 60795,' index 56437)

np.save('data/O16_w_event_keys', event_data)