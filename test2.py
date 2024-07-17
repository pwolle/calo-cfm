# %%
import matplotlib.pyplot as plt
import numpy as np

from data import preprocess

data = preprocess("data/raw/", "data/raw/*.h5")
data.max()


# %%
def sigmoid_inv(x):
    return np.log(x / (1 - x))


def pre(x):
    x = x / 300
    x = np.clip(x, 1e-5, 1 - 1e-5)
    x = sigmoid_inv(x)
    x = x + 6.5
    x = x / 2
    return x


x = data[: 1024 * 2]

x = x.flatten()

x1 = x[x > 0]
x2 = x[x <= 0]

# try replacing them by their convolution??

x1 = pre(x1)
print(x1.mean(), x1.std())

x2 = pre(x2)

# z = np.random.triangular(0, 2, 4, size=x2.size)
# x2 = x2 + np.random.laplace(0, 0.1, size=x2.size)
# z = np.random.normal(0, 10, size=x2.size)
# x2 += -z #- np.abs(z)
# x2 = x2 - 20

_, bins, _ = plt.hist([x1, x2], bins=64, stacked=True)
plt.yscale("log")
# plt.xscale("symlog", linthresh=1e-8)

xticks = np.linspace(bins.min(), bins.max(), 10)
# xlabels =

# plt.xlim(0, 150)
plt.show()

bins.max()

# %%
# np.count_nonzero(x == x.min()) / x.size

shower = data[0]

for i in range(0, 45, 9):
    x = shower[..., i]
    x = x + x.mean(-1, keepdims=True) * 1e-8

    plt.matshow(pre(x))
    plt.colorbar()
    plt.show()
