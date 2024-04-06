import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(1)


class network():
    def __init__(self):
        self.L = np.empty

    def Laplacian(self) -> np.ndarray:
        # Set graph Laplacian matrix
        a = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ])
        A = a + a.T
        D = np.diag(np.sum(A, axis=0))
        L = D - A

        return L


def Consensus(dimention: int, step: int) -> np.ndarray:
    G = network()
    L = G.Laplacian()
    n = L.shape[0]
    x = np.zeros((n, dimention, step))  # xの形状を修正
    x[:, :, 0] = np.random.uniform(size=(n, dimention))
    for k in range(1, step):
        x[:, :, k] = x[:, :, k - 1] - 0.05 * L @ x[:, :, k - 1]
    return x


def static_figure() -> None:
    x = Consensus(2, 300)
    plt.rcParams["font.size"] = 24
    fig1 = plt.figure(figsize=(16, 12), tight_layout=True)
    ax1 = fig1.add_subplot(211)
    for i in range(x.shape[0]):
        ax1.plot(x[i, 0, :], marker='o', markersize=1, label='Agent {}'.format(i))
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    ax1.set_title("Control Input for x-axis")
    ax2 = fig1.add_subplot(212)
    for i in range(x.shape[0]):
        ax2.plot(x[i, 1, :], marker='o', markersize=1, label='Agent {}'.format(i))
    ax2.set_title("Control Input for y-axis")
    fig1.savefig('Result.png')


def RenderGIF() -> None:
    x = Consensus(2, 300)
    plt.rcParams["font.size"] = 24
    fig2 = plt.figure(figsize=(12, 12), tight_layout=True)
    ax3 = fig2.add_subplot(111)
    ax3.set_xlim(min(x[:, 0, 0]) - 0.1, max(x[:, 0, 0]) + 0.1)
    ax3.set_ylim(min(x[:, 1, 0]) - 0.1, max(x[:, 1, 0]) + 0.1)
    ax3.set_title("Agent Trajectory")
    scatter_plots = [ax3.scatter(x[i, 0, 0], x[i, 1, 0], marker='o', s=100) for i in range(x.shape[0])] 

    def update(frame):
        for i in range(x.shape[0]):
            scatter_plots[i].set_offsets(np.column_stack([x[i, 0, frame], x[i, 1, frame]]))
        return scatter_plots

    anim = animation.FuncAnimation(fig2, update, frames=x.shape[2], blit=True)
    anim.save('Result.gif', writer='pillow', fps=40)


if __name__ == '__main__':
    RenderGIF()
    static_figure()
