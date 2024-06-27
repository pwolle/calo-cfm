# %%
import einx
import flarejax as fj
import flarenet as fn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import numpy as onp
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped


def to_patches(voxels, bx, by, bz) -> jax.Array:
    return einx.rearrange(  # type: ignore
        "(x bx) (y by) (z bz) -> x y z (bx by bz)",
        voxels,
        bx=bx,
        by=by,
        bz=bz,
    )


def get_coordinates(r, t, z):
    r = jnp.linspace(0, 1, r)
    t = jnp.linspace(0, 2 * jnp.pi, t)
    z = jnp.linspace(-1, 1, z)

    R, T, Z = jnp.meshgrid(r, t, z, indexing="ij")

    X = R * jnp.cos(T)
    Y = R * jnp.sin(T)

    return jnp.stack([X, Y, Z], axis=-1)


def to_voxels(patches, bx, by, bz) -> jax.Array:
    return einx.rearrange(  # type: ignore
        "x y z (bx by bz) -> (x bx) (y by) (z bz)",
        patches,
        bx=bx,
        by=by,
        bz=bz,
    )


def fourier_embedding(x, n):
    assert n % 2 == 0
    n = n // 2

    x = x[..., None]
    x = x * jnp.pi * jnp.arange(1, n + 1) / 2

    s = jnp.sin(x)
    c = jnp.cos(x)

    return jnp.concatenate([s, c], axis=-1)


class Embedding(fj.Module, replace=True):
    __module_name = "calo-cfm.Embedding"

    linear: fn.Linear

    dim_fourier: int = fj.field(static=True)
    dim_patches: tuple[int, int, int] = fj.field(static=True)

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        dim: int,
        dim_fourier: int,
        dim_patches: tuple[int, int, int],
    ):
        dim_in = onp.prod(dim_patches) + dim_fourier * (len(dim_patches) + 1)
        dim_in = int(round(dim_in, 0))

        return cls(
            linear=fn.Linear.init(key, dim_in, dim),
            dim_fourier=dim_fourier,
            dim_patches=dim_patches,
        )

    def __call__(self, x: jax.Array, t) -> jax.Array:
        assert x.ndim == 3
        x = to_patches(x, *self.dim_patches)

        t = fourier_embedding(t, self.dim_fourier)
        t = jnp.tile(t, x.shape[:-1] + (1,))

        c = get_coordinates(*x.shape[:-1])
        c = jax.vmap(fourier_embedding, (-1, None), -1)(c, self.dim_fourier)

        c = c.reshape(c.shape[:-2] + (-1,))
        x = jnp.concat([x, c, t], axis=-1)

        return self.linear(x)

    def to_voxels(self, patches):
        return to_voxels(patches, *self.dim_patches)


@jaxtyped(typechecker=fj.typecheck)
def attention_matrix(
    q: Float[Array, "*b q d"],
    k: Float[Array, "*b k d"],
) -> Float[Array, "*b q k"]:
    q = q * jnp.sqrt(q.shape[-1])
    a = jnp.einsum("...qd,...kd->...qk", q, k)
    a = jnn.softmax(a, axis=-1)
    return a


@jaxtyped(typechecker=fj.typecheck)
def attention(
    q: Float[Array, "*b q d"],
    k: Float[Array, "*b k d"],
    v: Float[Array, "*b k d"],
) -> Float[Array, "*b q d"]:
    a = attention_matrix(q, k)
    a = jnp.einsum("...qk,...kd->...qd", a, v)
    return a


class MultiHeadAttention(fj.Module, replace=True):
    __module_name = "calo-cfm.Attention"

    watt: fn.Linear
    norm: fn.LayerNorm
    dim_att: int = fj.field(static=True)

    @classmethod
    def init(cls, key: PRNGKeyArray, dim, dim_att):
        if dim % dim_att != 0:
            error = f"dim {dim} must be divisible by nheads {dim_att}"
            raise ValueError(error)

        return cls(
            watt=fn.Linear.init(
                key,
                dim,
                dim * 3,
                use_bias=False,
            ),
            norm=fn.LayerNorm.init(dim),
            dim_att=dim_att,
        )

    @property
    def nheads(self):
        return self.watt.dim // self.dim_att // 3

    @property
    def dim(self):
        return self.watt.dim // 3

    @jaxtyped(typechecker=fj.typecheck)
    def __call__(
        self, x: Float[Array, "*b q {self.dim}"]
    ) -> Float[Array, "*b q {self.dim}"]:
        x = self.norm(x)

        qkv = self.watt(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(q.shape[:-1] + (self.nheads, self.dim_att))
        k = k.reshape(k.shape[:-1] + (self.nheads, self.dim_att))
        v = v.reshape(v.shape[:-1] + (self.nheads, self.dim_att))

        r = attention(q, k, v)
        r = r.reshape(r.shape[:-2] + (self.nheads * self.dim_att,))

        return r + x


def transformer_block(key, dim, dim_att):
    key1, key2, key3 = jrandom.split(key, 3)
    return fn.Sequential(
        (
            MultiHeadAttention.init(key1, dim, dim_att),
            fn.LayerNorm.init(dim),
            fn.Linear.init(key2, dim, dim * 4),
            fn.GELU(),
            fn.Linear.init(key3, dim * 4, dim),
        )
    )


class Transformer(fj.Module, replace=True):
    __module_name = "calo-cfm.Transformer"

    inputs: Embedding
    blocks: fj.Sequential
    output: fn.Linear

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        dim: int,
        dim_att: int,
        dim_patches: tuple[int, int, int],
        nblocks: int,
    ):
        key_e, key_b, key_o = jrandom.split(key, 3)
        inputs = Embedding.init(key_e, dim, dim_att, dim_patches)

        blocks = []
        for key_b in jrandom.split(key_b, nblocks):
            block = transformer_block(key_b, dim, dim_att)
            blocks.append(block)

        blocks = fj.Sequential(tuple(blocks))

        output = fn.Linear.init(key_o, dim, int(onp.prod(dim_patches)))
        return cls(inputs=inputs, blocks=blocks, output=output)

    def __call__(self, x: jax.Array, t: jax.Array) -> jax.Array:
        x = self.inputs(x, t)

        x_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])

        x = self.blocks(x)
        x = self.output(x)

        x = x.reshape(*x_shape, -1)
        x = self.inputs.to_voxels(x)

        return x


def main():

    key = jrandom.PRNGKey(0)
    model = Transformer.init(key, 32, 4, (3, 2, 5), 1)

    x = jnp.arange(9 * 16 * 45).reshape((9, 16, 45))
    t = jnp.ones((1, 1, 1))

    y = model(x, t)

    print(y.shape)


if __name__ == "__main__":
    main()
