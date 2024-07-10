# %%
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def explode(data):
    size = np.array(data.shape) * 2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e


def linspace_with_gaps(start, stop, num, gap=0):
    lin = np.linspace(start, stop - gap, num // 2)

    dif = lin[1] - lin[0]
    with_gap = np.zeros(num)

    with_gap[0::2] = lin
    with_gap[1::2] = lin + dif - gap

    return with_gap


def plot_shower(shower):
    shower = shower - np.min(shower)
    shower = shower / np.max(shower)
    shower = explode(shower)

    r_dim, thet_dim, z_dim = shower.shape

    r_grid = linspace_with_gaps(0, 1, r_dim + 1)
    theta_grid = linspace_with_gaps(0, 2 * np.pi, thet_dim + 1)
    z_grid = linspace_with_gaps(-1, 1, z_dim + 1)

    R, Theta, Z = np.meshgrid(r_grid, theta_grid, z_grid, indexing="ij")

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    hsv_colors = np.zeros(shower.shape + (3,))
    hsv_colors[..., 0] = Theta[:-1, :-1, :-1] / (2 * np.pi)
    hsv_colors[..., 1] = 1
    hsv_colors[..., 2] = 1

    rgb_colors = matplotlib.colors.hsv_to_rgb(hsv_colors)

    colors = np.zeros(shower.shape + (4,))
    colors[..., :3] = rgb_colors

    alpha = shower
    colors[..., 3] = alpha

    fig = plt.figure(figsize=(6, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    # Use ax.voxels to plot the voxels
    print(np.mean(alpha))
    ax.voxels(  # type: ignore
        X,
        Y,
        Z,
        alpha > 0.01,
        facecolors=colors,
        shade=False,
    )

    # aspect the same
    ax.set_box_aspect([1, 1, 1])

    # no ticks
    ax.set_xticks([-1, 0, 1], [])
    ax.set_yticks([-1, 0, 1], [])
    ax.set_zticks([-1, 0, 1], [])  # type: ignore

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)  # type: ignore

    ax.set_xlabel("X", labelpad=-10)
    ax.set_ylabel("Y", labelpad=-10)
    ax.set_zlabel("Z", labelpad=-10)  # type: ignore


def main():
    import random
    from _preprocess import preprocess

    showers = preprocess("raw/", "raw/*.h5")
    i = random.randint(0, len(showers))
    shower = showers[i]

    plot_shower(shower[..., ::5])
    plt.show()


if __name__ == "__main__":
    main()


# %%
