# %%
import jax.random as jrandom
import jax.numpy as jnp
import flarejax as fj

from data._visual import plot_shower, explode
from data._preprocess import elementwise_preprocess_inv
import models
from data import preprocess
import matplotlib.pyplot as plt

from tqdm import trange

# %%

model = fj.load("model.npz")
model = fj.Jit(model)

key = jrandom.PRNGKey(0)

x = jrandom.normal(key, (1, 9, 16, 45))
t = jnp.zeros((1,))

n = 1024

for _ in trange(n):
    v = model(x, t) * 3
    # print(v)

    x = x + v * 1 / n
    t = t + 1 / n

# x = elementwise_preprocess_inv(x)
plot_shower(explode(x[0]))

plt.show()
plt.hist(x[0].flatten())
plt.show()


# %%

data = preprocess("data/raw/", "data/raw/*.h5")
plot_shower(explode(data[7]))

plt.show()
plt.hist(data[7].flatten())
plt.show()

# %%
