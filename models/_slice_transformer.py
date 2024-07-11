import flarejax as fj
import flarenet as fn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


def fourier_encode(x, dim: int):
    assert dim % 2 == 0

    r = jnp.linspace(0, 1, dim // 2)

    r = (2 * jnp.pi * 1_000) ** r
    x = x[..., None] * r

    s = jnp.sin(x)
    c = jnp.cos(x)

    return jnp.concatenate([s, c], axis=-1)


class InputBlock(fj.Module, replace=True):
    __module_name = "_channel_mixer.InputBlock"

    linear: fn.Linear

    @classmethod
    def init(cls, key: PRNGKeyArray, channels: int):
        assert channels % 2 == 0
        return cls(linear=fn.Linear.init(key, channels, channels))

    def __call__(self, x, t):
        t = jnp.tile(t, x.shape)
        t = fourier_encode(t, self.linear.dim // 2)
        y = fourier_encode(x, self.linear.dim // 2)
        z = jnp.concatenate([y, t], axis=-1)
        return self.linear(z)
