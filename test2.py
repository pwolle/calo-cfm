# %%
from data import preprocess

data = preprocess("data/raw/", "data/raw/*.h5")

# %%
import matplotlib.pyplot as plt
import random

i = random.randint(0, len(data))

# R, T, H

for i in range(45)[::9]:
    plt.matshow(data[i, :, :, i])
    plt.show()

# %%
data.shape
# %%
9 * 16
