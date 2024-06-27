import functools
import os

import flarejax as fj
import flarenet as fn
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import optax
import tqdm
from jaxtyping import PRNGKeyArray

from jax.lib import xla_bridge

print(xla_bridge.get_backend().platform)


@functools.partial(jax.jit, static_argnames=("batch_size",))
def sample_source(key: PRNGKeyArray, batch_size: int = 1024) -> jax.Array:
    return jrandom.normal(key, (batch_size, 1))


@functools.partial(jax.jit, static_argnames=("batch_size",))
def sample_target(key, batch_size: int = 1024) -> jax.Array:
    key_norm, key_mode = jrandom.split(key)
    norm = jrandom.normal(key_norm, (batch_size, 1))

    mode = jrandom.uniform(key_mode, (batch_size, 1))
    mode = (mode > 0.5) * 2.5 - 1.25

    return mode + norm * 0.5


def low_descrepancy_uniform(key: PRNGKeyArray, n: int) -> jax.Array:
    l = jnp.linspace(0, 1, n, endpoint=False)
    z = jrandom.uniform(key, (n,))
    return l + z / n


@functools.partial(jax.jit, static_argnames=("batch_size",))
def ot_cfm_loss(key: PRNGKeyArray, model, batch_size: int = 1024):
    key_source, key_target, key_time = jrandom.split(key, 3)

    x0 = sample_source(key_source, batch_size).sort()
    x1 = sample_target(key_target, batch_size).sort()

    t = low_descrepancy_uniform(key_time, batch_size)[..., None]
    xt = x1 * t + x0 * (1 - t)
    ut = x1 - x0

    xh = model(xt, t)
    return ((xh - ut) ** 2).mean()


class Model(fj.Module, register=False):
    modules: fj.Sequential

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        dim: int = 128,
        num_layers: int = 3,
    ):
        key, key1, key2 = jrandom.split(key, 3)
        layers = []

        layer = fn.Linear.init(key1, 2, dim)
        layers.append(layer)
        layers.append(fn.GELU())

        for key in jrandom.split(key2, num_layers):
            layer = fn.Linear.init(key, dim, dim)
            layers.append(layer)
            layers.append(fn.GELU())

        layer = fn.Linear.init(key2, dim, 1)
        layers.append(layer)

        return cls(fj.Sequential(layers))

    def __call__(self, x, t):
        x = jnp.concatenate([x, t], axis=-1)
        return self.modules(x)


key = jrandom.PRNGKey(0)
key, key_model = jrandom.split(key)

model = Model.init(key_model, dim=32, num_layers=3)
model = fj.Jit(model)

opt = optax.adam(3e-4)
opt_state = opt.init(model)  # type: ignore


@jax.jit
def train_step(key, model, opt_state):
    key, key_batch = jrandom.split(key)

    loss, grad = jax.value_and_grad(ot_cfm_loss, 1)(key_batch, model, 1024 * 4)
    updates, opt_state = opt.update(grad, opt_state)

    model = optax.apply_updates(model, updates)
    return loss, model, opt_state


losses = []
print("training")

for i in tqdm.trange(10000):
    key, key_batch = jrandom.split(key)
    loss, model, opt_state = train_step(key_batch, model, opt_state)
    losses.append(loss)

assert isinstance(model, fj.Jit)

n = 1024

key, subkey = jrandom.split(key)
x = sample_source(subkey, 1024 * 32)
t = jnp.zeros((x.shape[0], 1))

print("sampling")

for i in tqdm.trange(n):
    h = model(x, t)
    t = t + 1 / n
    x = x + model(x, t) / n


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 6))

key, key_source, key_target = jrandom.split(key, 3)

source_samples = sample_source(key_source, 1024 * 128)
target_samples = sample_target(key_target, 1024 * 128)

bins = 512

_, bins, _ = ax1.hist(
    source_samples,
    bins=bins,
    density=True,
    label="source",
    histtype="step",
    color="gray",
    alpha=0.5,
)

_, bins, _ = ax1.hist(
    x,
    bins=bins,
    density=True,
    histtype="step",
    label="model",
    color="black",
)

_, bins, _ = ax1.hist(
    target_samples,
    bins=bins,
    density=True,
    label="target",
    histtype="step",
    color="red",
)

ax1.set_xlim(-3.5, 3.5)

ax1.set_xlabel("x")
ax1.set_ylabel("density")

ax1.legend(frameon=False)
ax1.tick_params(which="both", direction="in", top=True, right=True)
ax1.minorticks_on()

ax2.plot(losses)
ax2.set_yscale("log")

ax2.set_xlim(0, len(losses))

ax2.set_xlabel("iteration")
ax2.set_ylabel("loss")

ax2.tick_params(which="both", direction="in", top=True, right=True)
ax2.minorticks_on()

os.makedirs("plots", exist_ok=True)
fig.savefig("plots/ot_cfm.pdf", bbox_inches="tight", transparent=True)
