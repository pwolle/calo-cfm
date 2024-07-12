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


def fourier_encode(x, dim: int):
    assert dim % 2 == 0

    r = jnp.linspace(0, 1, dim // 2)

    r = (2 * jnp.pi * 1_000) ** r
    x = x[..., None] * r

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

        t = fourier_encode(t, self.dim_fourier)
        t = jnp.tile(t, x.shape[:-1] + (1,))

        c = get_coordinates(*x.shape[:-1])
        c = jax.vmap(fourier_encode, (-1, None), -1)(c, self.dim_fourier)

        c = c.reshape(c.shape[:-2] + (-1,))
        x = jnp.concat([x, c, t], axis=-1)

        return self.linear(x)

    def to_voxels(self, patches):
        return to_voxels(patches, *self.dim_patches)


class AttentionBlock(fj.Module, replace=True):
    __module_name = "_transformer2.AttentionBlock"

    wqkv: fn.Linear
    wout: fn.Linear
    norm: fn.LayerNorm

    nheads: int = fj.field(static=True)

    @classmethod
    def init(cls, key: PRNGKeyArray, dim: int, nheads: int):
        assert dim % nheads == 0

        key_qkv, key_out = jrandom.split(key)
        return cls(
            wqkv=fn.Linear.init(key_qkv, dim, 3 * dim),
            wout=fn.Linear.init(key_out, dim, dim),
            norm=fn.LayerNorm.init(dim),
            nheads=nheads,
        )

    def __call__(
        self,
        x: Float[Array, "seq {self.dim}"],
    ) -> Float[Array, "seq {self.dim}"]:
        x = r = x

        qkv = self.wqkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = q.reshape(q.shape[:-1] + (self.nheads, -1))
        k = k.reshape(k.shape[:-1] + (self.nheads, -1))
        v = v.reshape(v.shape[:-1] + (self.nheads, -1))

        q = q / jnp.sqrt(q.shape[-1])
        A = jnp.einsum("qhd, khd -> hqk", q, k)

        A = jnn.softmax(A, axis=-1)

        x = jnp.einsum("hqk, khd -> qhd", A, v)
        x = x.reshape(x.shape[:-2] + (self.dim,))

        x = self.wout(x)
        x = (x + r) * 2**-0.5
        x = self.norm(x)
        return x

    @property
    def dim(self) -> int:
        return self.wout.dim


class MLPBlock(fj.Module, replace=True):
    __module_name = "_transformer2.MLPBlock"

    w1: fn.Linear
    w2: fn.Linear

    n1: fn.LayerNorm
    n2: fn.LayerNorm

    @classmethod
    def init(cls, key: PRNGKeyArray, dim: int):
        key1, key2 = jrandom.split(key)
        return cls(
            w1=fn.Linear.init(key1, dim, dim * 2),
            w2=fn.Linear.init(key2, dim * 2, dim),
            n1=fn.LayerNorm.init(dim),
            n2=fn.LayerNorm.init(dim),
        )

    @jaxtyped(typechecker=fj.typecheck)
    def __call__(
        self,
        x: Float[Array, "*b {self.dim}"],
    ) -> Float[Array, "*b {self.dim}"]:
        r = x
        x = self.n1(x)
        x = self.w1(x)
        x = jnn.gelu(x, approximate=True)
        x = self.w2(x)
        x = (x + r) * 2**-0.5
        x = self.n2(x)
        return x

    @property
    def dim(self) -> int:
        return self.w2.dim


class Transformer(fj.Module, replace=True):
    __module_name = "_transformer2.Transformer"

    inputs: Embedding
    blocks: fj.Sequential
    output: fn.Linear

    @classmethod
    def init(
        cls,
        key: PRNGKeyArray,
        dim: int,
        dim_fourier: int,
        dim_patches: tuple[int, int, int],
        nheads: int,
        nblocks: int,
    ):
        key_e, key_b, key_o = jrandom.split(key, 3)
        inputs = Embedding.init(key_e, dim, dim_fourier, dim_patches)

        blocks = []
        for key_b in jrandom.split(key_b, nblocks):
            key_att, key_mlp = jrandom.split(key_b)

            block = AttentionBlock.init(key_att, dim, nheads)
            blocks.append(block)

            block = MLPBlock.init(key_mlp, dim)
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
    model = Transformer.init(
        key,
        dim=32,
        dim_fourier=4,
        dim_patches=(3, 2, 5),
        nheads=4,
        nblocks=2,
    )

    x = jnp.arange(9 * 16 * 45).reshape((9, 16, 45))
    t = jnp.ones(())

    y = model(x, t)

    print(y.shape)


if __name__ == "__main__":
    main()
