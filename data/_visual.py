# %%
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_shower(shower):
    r_dim, thet_dim, z_dim = shower.shape

    r_grid = np.linspace(0, 1, r_dim + 1)
    theta_grid = np.linspace(0, 2 * np.pi, thet_dim + 1)
    z_grid = np.linspace(-1, 1, z_dim + 1)

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
    alpha = alpha - np.min(shower)
    alpha = alpha / np.max(alpha)
    colors[..., 3] = alpha

    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    # Use ax.voxels to plot the voxels
    ax.voxels(  # type: ignore
        X,
        Y,
        Z,
        shower > 0.01,
        facecolors=colors,
        edgecolors=colors,
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

    import h5py

    path = "raw/ddsim_mesh_Par04_gamma_500events_1GeV1TeV_GPSflat_edm4hep_12388001_part1.h5"

    with h5py.File(path, "r") as f:
        showers = f["showers"]  # type: ignore
        i = random.randint(0, len(showers))  # type: ignore
        shower: np.ndarray = showers[i]  # type: ignore

    plot_shower(shower=shower)  # [::2, ::2, ::2])
    plt.show()


if __name__ == "__main__":
    main()


# %%
