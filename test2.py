# %%
from data import preprocess

data = preprocess("data/raw/", "data/raw/*.h5")

# %%
import matplotlib.pyplot as plt
import random


x = data[: 1024 * 32]
x = x.flatten()

x = x[x > 0]

plt.hist(x, bins=100)
plt.show()
