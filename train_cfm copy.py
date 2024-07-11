# %%
import flarejax as fj
import jax
import jax.numpy as jnp
import jax.random as jrandom
import memmpy
import optax
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

from data import preprocess
from models import ChannelMixer, Transformer

import wandb
import datetime


def main(
    batch_size,
    seed,
    learning_rate,
    channels,
    depth,
):
    def spaced_uniform(key: PRNGKeyArray, n: int) -> jax.Array:
        l = jnp.linspace(0, 1, n, endpoint=False)
        z = jrandom.uniform(key, (n,))
        return l + z / n

    @jax.jit
    def cfm_loss(key: PRNGKeyArray, model, source, target):
        t = spaced_uniform(key, source.shape[0])[:, None, None, None]

        xt = target * t + source * (1 - t)
        ut = target - source

        xh = model(xt, t)
        return ((xh - ut) ** 2).mean()

    key = jrandom.PRNGKey(seed)
    key, key_model = jrandom.split(key)

    model = ChannelMixer.init(key_model, (9, 16, 45), channels, depth)
    # model = Transformer.init(key_model, 32, 4, (3, 2, 5), 1)
    model = fj.VMap(model)

    opt = optax.adam(learning_rate)
    opt_state = opt.init(model)  # type: ignore

    @jax.jit
    def train_step(key, model, data, opt_state):
        source = jrandom.normal(key, data.shape)
        loss, grad = jax.value_and_grad(cfm_loss, 1)(key, model, source, data)

        updates, opt_state = opt.update(grad, opt_state)

        model = optax.apply_updates(model, updates)
        return loss, model, opt_state

    data = preprocess("data/raw/", "data/raw/*.h5")

    shuffle = memmpy.shuffle_fast(data.shape[0], seed=seed)

    valid_indicies = memmpy.batch_indicies_split(
        data.shape[0],
        64,
        "valid",
        10,
        drop_remainder=False,
    )
    valid_indicies = next(iter(valid_indicies))
    valid_batch = data[shuffle(valid_indicies)]

    timestamp = datetime.datetime.now()
    timestamp = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    wandb.init(
        project="calo-cfm",
        config={
            "batch_size": batch_size,
            "seed": seed,
            "learning_rate": learning_rate,
            "channels": channels,
            "depth": depth,
            "time": timestamp,
        },
    )

    try:
        for _ in range(10):
            batch_indicies = memmpy.batch_indicies_split(
                data.shape[0],
                batch_size,
                "train",
                10,
                drop_remainder=True,
            )

            for indicies in tqdm(batch_indicies, total=int(len(data) * 0.8 / 32)):
                indicies = shuffle(indicies)
                batch = data[indicies]

                key, key_train = jrandom.split(key)
                loss, model, opt_state = train_step(key_train, model, batch, opt_state)

                key, key_valid = jrandom.split(key)
                loss_valid = cfm_loss(key_valid, model, valid_batch, valid_batch)

                wandb.log(
                    {
                        "loss": loss,
                        "loss_valid": loss_valid,
                    }
                )
    except KeyboardInterrupt:
        pass

    fj.save(f"model_{timestamp}.npz", model)

    # upload model
    # wandb.save(f"model_{timestamp}.npz")

    wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    args = parser.parse_args()

    main(
        batch_size=args.batch_size,
        seed=args.seed,
        learning_rate=args.learning_rate,
        channels=args.channels,
        depth=args.depth,
    )
