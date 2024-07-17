# %%
import datetime

import flarejax as fj
import jax
import jax.numpy as jnp
import jax.random as jrandom
import memmpy
import optax
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

import wandb
from data import preprocess
from models import Transformer


def sigmoid_inv(x):
    return jnp.log(x / (1 - x))


@jax.jit
def scale_data(x):
    x = x / 300
    x = jnp.clip(x, 1e-5, 1 - 1e-5)
    x = sigmoid_inv(x)
    x = x + 6.5
    x = x / 2
    return x.astype(jnp.float32)


def main(
    batch_size: int,
    seed: int,
    learning_rate: float,
    dim: int,
    dim_fourier: int,
    dim_patches: tuple[int, int, int],
    nheads: int,
    nblocks: int,
    nepochs: int,
):
    data = preprocess("data/raw/", "data/raw/*.h5")

    shuffle = memmpy.shuffle_fast(data.shape[0], seed=seed)

    valid_indicies = memmpy.batch_indicies_split(
        data.shape[0],
        128,
        "valid",
        10,
        drop_remainder=False,
    )
    valid_indicies = next(iter(valid_indicies))
    valid_batch = data[shuffle(valid_indicies)]

    def spaced_uniform(key: PRNGKeyArray, n: int) -> jax.Array:
        l = jnp.linspace(0, 1, n, endpoint=False)
        z = jrandom.uniform(key, (n,))
        return l + z / n

    @jax.jit
    def cfm_loss(key: PRNGKeyArray, model, target):
        target = scale_data(target)
        key_t, key_s = jrandom.split(key)

        source = jrandom.normal(key_s, target.shape)
        t = spaced_uniform(key_t, source.shape[0])[:, None, None, None]

        xt = target * t + source * (1 - t)
        ut = target - source

        xh = model(xt, t)
        return ((xh - ut) ** 2).mean()

    key = jrandom.PRNGKey(seed)
    key, key_model = jrandom.split(key)

    model = Transformer.init(
        key_model,
        dim=dim,
        dim_fourier=dim_fourier,
        dim_patches=dim_patches,
        nheads=nheads,
        nblocks=nblocks,
    )
    model = fj.VMap(model)

    epoch_steps = int(len(data) * 0.8 / batch_size)

    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=1e-6,
    #     peak_value=learning_rate,
    #     warmup_steps=100,
    #     decay_steps=epoch_steps * nepochs,
    #     end_value=3e-5,
    # )
    schedule = learning_rate

    opt = optax.adam(schedule)
    opt_state = opt.init(model)  # type: ignore

    @jax.jit
    def train_step(key, model, data, opt_state):
        loss, grad = jax.value_and_grad(cfm_loss, 1)(key, model, data)

        updates, opt_state = opt.update(grad, opt_state)

        model = optax.apply_updates(model, updates)
        return loss, model, opt_state

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="calo-cfm",
        config={
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate": learning_rate,
            "dim": dim,
            "dim_patches": dim_patches,
            "nheads": nheads,
            "nblocks": nblocks,
            "nepochs": nepochs,
            "timestamp": timestamp,
        },
    )

    try:
        step = 0

        for _ in range(nepochs):
            batch_indicies = memmpy.batch_indicies_split(
                data.shape[0],
                batch_size,
                "train",
                10,
                drop_remainder=True,
            )

            for indicies in tqdm(batch_indicies, total=epoch_steps):
                indicies = shuffle(indicies)
                batch = data[indicies]

                key, key_train = jrandom.split(key)
                loss, model, opt_state = train_step(key_train, model, batch, opt_state)

                if step % 100 == 0:
                    key, key_valid = jrandom.split(key)
                    loss_valid = cfm_loss(key_valid, model, valid_batch)

                    wandb.log(
                        {
                            "loss": loss,
                            "loss_valid": loss_valid,
                        },
                        step=step,
                    )

                step += 1
    except KeyboardInterrupt:
        fj.save(f"model_{timestamp}.npz", model)
        wandb.finish()
        exit()

    fj.save(f"model_{timestamp}.npz", model)

    wandb.save(f"model_{timestamp}.npz")
    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--dim_fourier", type=int, default=64)
    parser.add_argument("--dim_patches", type=int, nargs=3, default=[3, 4, 15])
    parser.add_argument("--nheads", type=int, default=16)
    parser.add_argument("--nblocks", type=int, default=8)
    parser.add_argument("--nepochs", type=int, default=5)
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        dim=args.dim,
        dim_fourier=args.dim_fourier,
        dim_patches=tuple(args.dim_patches),  # type: ignore
        nheads=args.nheads,
        nblocks=args.nblocks,
        nepochs=args.nepochs,
    )
