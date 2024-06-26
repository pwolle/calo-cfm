import flarejax as fj
import flarenet as fn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom


def make_mlp(key, dim, act) -> fn.Residual:
    key1, key2 = jrandom.split(key)
    layers = (
        fn.LayerNorm.init(dim),
        fn.Linear.init(key1, 1, dim),
        act(),
        fn.Linear.init(key2, dim, dim),
    )
    layers = fj.Sequential(layers)
    return fn.Residual(layers)


def circular_transpose(x):
    return x.transpose(-1, *range(x.ndim - 1))


class PixelMixer(fj.Module):
    __module_name = "calo-cfm.PixelMixer"

    layers_in: fj.ModuleSequence
    layers: fj.ModuleSequence
    layers_out: fj.ModuleSequence

    @classmethod
    def init(
        cls,
        key,
        dim_in: tuple[int, ...],
        dim: int,
        depth: int,
        act=fn.GELU,
    ):
        layers_in = []

        for d in dim_in[::-1]:
            key, subkey = jrandom.split(key)
            layer = fj.Sequential(
                (
                    fn.Linear.init(subkey, d, dim, False),
                    fn.GELU(),
                )
            )
            layers_in.append(layer)

        layers_in = fj.ModuleSequence(layers_in)

        layers = []
        for _ in range(depth * len(dim_in)):
            key, subkey = jrandom.split(key)
            layers.append(make_mlp(subkey, dim, act))

        layers = fj.ModuleSequence(layers)

        layers_out = []
        for d in dim_in[::-1]:
            key, subkey = jrandom.split(key)
            layers_out.append(fn.Linear.init(subkey, dim, d))

        layers_out = fj.ModuleSequence(layers_out)
        return cls(layers_in, layers, layers_out)

    def __call__(self, x):
        for layer_in in self.layers_in:
            x = layer_in(x)
            x = circular_transpose(x)

        for layer in self.layers:
            x = layer(x)
            x = circular_transpose(x)

        for layer_out in self.layers_out:
            x = layer_out(x)
            x = circular_transpose(x)

        return x


def main():
    key = jrandom.PRNGKey(0)

    x = jnp.zeros((2, 3, 5))
    model = PixelMixer.init(key, x.shape, 32, 0)

    # print(model)
    y = jax.eval_shape(model, x)
    # print(y.shape)


if __name__ == "__main__":
    main()
