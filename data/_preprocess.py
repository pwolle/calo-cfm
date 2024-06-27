# %%
import glob
import hashlib
import os

import h5py
import memmpy
import numba
import numpy as np


@numba.jit(nopython=True)
def elementwise_preprocess(x, k=4):
    for _ in range(k):
        x = np.log(x + 1)

    x = x * (2 / 1.038)
    x = x - 1
    return x


def elementwise_preprocess_inv(x, k=4):
    x = x + 1
    x = x * (1.038 / 2)

    for _ in range(k):
        x = np.exp(x) - 1

    return x


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
        return array.reshape(-1, *shape)

    vector = memmpy.Vector()

    for path in glob.glob(read_path):
        with h5py.File(path, "r") as f:
            showers = f["showers"]
            n = len(showers)  # type: ignore

            for s, e in memmpy.batch_slices(n, 1024 * 32, False):
                x = showers[s:e]  # type: ignore
                x = elementwise_preprocess(x)

                print(e)
                vector.extend(x)

    vector.save(write_file)
    return preprocess(write_path, read_path)


# preprocess("raw", "raw/*.h5").shape[0] / 1e6
