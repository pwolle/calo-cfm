# %%
import flarejax as fj
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import memmpy
import optax
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

from data import preprocess
from models import Transformer
import numpy as onp


jax.config.update("jax_debug_nans", True)


def spaced_uniform(key: PRNGKeyArray, n: int) -> jax.Array:
    l = jnp.linspace(0, 1, n, endpoint=False)
    z = jrandom.uniform(key, (n,))
    return l + z / n


def cfm_loss(key: PRNGKeyArray, model, source, target):
    t = spaced_uniform(key, source.shape[0])[:, None, None, None]

    xt = target * t + source * (1 - t)
    ut = target - source

    xh = model(xt[0], t[0])
    exit()

    return ((xh - ut) ** 2).mean()


key = jrandom.PRNGKey(0)
key, key_model = jrandom.split(key)

model = Transformer.init(key_model, 32, 4, (3, 2, 5), 1)
# model = fj.VMap(model)

opt = optax.adam(3e-4)
opt_state = opt.init(model)  # type: ignore


# @jax.jit
def train_step(key, model, data, opt_state):
    source = jrandom.normal(key, data.shape)
    loss = cfm_loss(key, model, source, data)
    # loss, grad = jax.value_and_grad(cfm_loss, 1)(key, model, source, data)

    updates, opt_state = opt.update(grad, opt_state)

    model = optax.apply_updates(model, updates)
    return loss, model, opt_state


data = preprocess("data/raw/", "data/raw/*.h5")

batch_indicies = memmpy.batch_indicies_split(data.shape[0], 32, "train", 10)
shuffle = memmpy.shuffle_fast(data.shape[0], seed=42)

losses = []

for i, indicies in enumerate(tqdm(batch_indicies, total=len(data) // 32)):
    indicies = shuffle(indicies)
    batch = data[indicies]
    batch = onp.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)

    loss, model, opt_state = train_step(key, model, batch, opt_state)
    losses.append(loss)

    if i % 100 == 0:
        print(loss)


fj.save("model.npz", model)

plt.plot(losses)
plt.yscale("log")
plt.show()

# %%
