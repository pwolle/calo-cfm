# %%
import glob
import hashlib
import os

import h5py
import memmpy
import numpy as np
from tqdm import tqdm


def hash_glob(path):
    hasher = hashlib.sha256()

    for path in glob.glob(path):
        path = os.path.abspath(path)

        hasher.update(path.encode())
        hasher.update(str(os.path.getmtime(path)).encode())

    return hasher.hexdigest()


def preprocess(write_path, read_path):
    os.makedirs(write_path, exist_ok=True)

    hashed = hash_glob(read_path)
    write_file = os.path.join(write_path, f"showers_{hashed}.npy")

    if os.path.exists(write_file):
        print("Found cashed file at ", write_file)

        with h5py.File(glob.glob(read_path)[0], "r") as f:
            shape = f["showers"].shape[1:]  # type: ignore

        array = np.memmap(write_file, dtype=np.float32, mode="r")
        array = array.reshape(-1, *shape)
        print(array.shape)
        return array

    vector = memmpy.Vector()

    for path in tqdm(glob.glob(read_path)):
        with h5py.File(path, "r") as f:
            showers = f["showers"]
            n = len(showers)  # type: ignore

            for s, e in memmpy.batch_slices(n, 1024 * 8, False):
                x = showers[s:e]  # type: ignore

                x = np.maximum(x, 0)
                x = np.log(1 + x) / (5.25 / 2)
                x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)

                x = x - 1
                x = np.clip(x, -1, 1)
                x = x.astype(np.float32)

                vector.extend(x)
                break

            break

    vector.save(write_file)
    return preprocess(write_path, read_path)


if __name__ == "__main__":
    preprocess("raw", "raw/*.h5")
