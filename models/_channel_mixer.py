# %%
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


class MLPBlock(fj.Module, replace=True):
    __module_name = "_channel_mixer.MLPBlock"

    linear1: fn.Linear
    linear2: fn.Linear

    norm1: fn.LayerNorm
    norm2: fn.LayerNorm

    @classmethod
    def init(cls, key: PRNGKeyArray, dim: int, dim_hidden: int | None = None):
        if dim_hidden is None:
            dim_hidden = dim * 2

        key1, key2 = jrandom.split(key)
        return cls(
            linear1=fn.Linear.init(key1, dim, dim_hidden),
            linear2=fn.Linear.init(key2, dim_hidden, dim),
            norm1=fn.LayerNorm.init(dim),
            norm2=fn.LayerNorm.init(dim),
        )

    @jaxtyped(typechecker=fj.typecheck)
    def __call__(
        self,
        x: Float[Array, "*b {self.dim}"],
    ) -> Float[Array, "*b {self.dim}"]:
        r = x

        x = self.norm1(x)
        x = self.linear1(x)

        x = jnn.gelu(x, approximate=True)
        x = self.linear2(x)

        x = (x + r) * 2**-0.5
        x = self.norm2(x)
        return x

    @property
    def dim(self):
        return self.linear2.dim


class MixerBlock(fj.Module, replace=True):
    __module_name = "_channel_mixer.MixerBlock"

    spatial: MLPBlock
    channel: MLPBlock

    axis: int = fj.field(static=True)

    @classmethod
    def init(cls, key: PRNGKeyArray, axis: int, spatial: int, channels: int):
        key_spatial, key_channel = jrandom.split(key)

        return cls(
            spatial=MLPBlock.init(key_spatial, spatial, channels * 2),
            channel=MLPBlock.init(key_channel, channels),
            axis=axis,
        )

    def __call__(self, x: jax.Array):
        x = x.swapaxes(-1, self.axis)
        x = self.spatial(x)
        x = x.swapaxes(-1, self.axis)
        x = self.channel(x)
        return x


class ChannelMixer(fj.Module, replace=True):
    __module_name = "_channel_mixer.ChannelMixer"

    inputs_block: InputBlock
    output_block: fn.Linear

    mixer_blocks: fj.ModuleSequence

    @classmethod
    def init(
        cls,
        key,
        dim_in: tuple[int, ...],
        channels: int,
        depth: int,
    ):
        key_inputs, key_mixer, key_outputs = jrandom.split(key, 3)
        inputs_block = InputBlock.init(key_inputs, channels)

        mixer_blocks = []

        for a, d in list(enumerate(dim_in)) * depth:
            key_mixer, subkey = jrandom.split(key_mixer)
            mixer_blocks.append(MixerBlock.init(subkey, a, d, channels))

        mixer_blocks = fj.ModuleSequence(mixer_blocks)

        output_block = fn.Linear.init(key_outputs, channels, 1)
        return cls(
            inputs_block=inputs_block,
            output_block=output_block,
            mixer_blocks=mixer_blocks,
        )

    def __call__(self, x, t):
        x = self.inputs_block(x, t)

        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        return self.output_block(x)[..., 0]


def main():
    key = jrandom.PRNGKey(0)
    # layer = InputBlock.init(key, 128)
    layer = ChannelMixer.init(key, (9, 16, 45), 8, 2)
    layer = fj.VMap(layer)

    x = jnp.zeros((32, 9, 16, 45))
    t = jnp.zeros((32,))
    y = layer(x, t)

    print(y.shape)


if __name__ == "__main__":
    main()

# %%
