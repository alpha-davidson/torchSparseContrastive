import h5py
import numpy as np
import tqdm

INPUT = "/data/46Ar/point_clouds/experimental/clean_run_0130.h5"
OUTPUT_BASE = "data/Ar46"

with h5py.File(INPUT, "r") as data:
    root = data["clean"]
    keys = list(root.keys())

    print(f"Found {len(keys)} events")

    event_lens = np.array(
        [root[key].shape[0] for key in keys],
        dtype=int,
    )
    np.save(OUTPUT_BASE + "_event_lens.npy", event_lens)

    max_hits = event_lens.max(initial=0)
    event_data = np.zeros((len(keys), max_hits, 6), dtype=float)

    for n, key in enumerate(tqdm.tqdm(keys)):
        event = root[key][...]

        if event.ndim != 2 or event.shape[1] < 6:
            raise ValueError(
                f"Event {key!r} has shape {event.shape}; "
                "expected (number_of_hits, at least 6)"
            )

        event_data[n, :len(event), 0] = event[:, 0]  # x
        event_data[n, :len(event), 1] = event[:, 1]  # y
        event_data[n, :len(event), 2] = event[:, 2]  # z
        event_data[n, :len(event), 3] = event[:, 4]  # charge
        event_data[n, :len(event), 4] = event[:, 5]  # NND
        event_data[n, :len(event), 5] = n            # event index

np.save(OUTPUT_BASE + "_w_event_keys.npy", event_data)

print("Output shape:", event_data.shape)
print(f"Output size: {event_data.nbytes / 1e9:.2f} GB")