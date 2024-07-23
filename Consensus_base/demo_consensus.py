import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

np.random.seed(1) # Set random seed


class network():
    def __init__(self):
        self.L = np.empty

    def Laplacian(self) -> np.ndarray:
        def GridAdjMatrix(N):
            n = int(np.sqrt(N))
            if n**2 != N:
                raise ValueError(f'the square root is not a natural number: sqrt({N}) = {np.sqrt(N)}')
            
            a = np.zeros((N, N))
            for i in range(N):
                if (i + 1) % n != 0 and i + 1 < N:
                    a[i, i + 1] = 1
                if i + n < N:
                    a[i, i + n] = 1

            weight = 0.01 + (2 - 0.01) * np.random.rand(N, N)
            a = weight * a
            A = a + a.T
            return A

        A = GridAdjMatrix(25)
        D = np.diag(np.sum(A, axis=0))
        L = D - A

        return L


def Consensus(dimension: int, step: int) -> (np.ndarray, np.ndarray):
    G = network()
    L, A = G.Laplacian()
    n = L.shape[0] # Get the number of nodes from matrix shape
    x = np.zeros((n, dimension, step))
    x[:, :, 0] = np.random.uniform(size=(n, dimension)) # Generate the state vector with random function
    for k in range(1, step):
        x[:, :, k] = x[:, :, k - 1] - 0.05 *  L @ x[:, :, k - 1] # Calculate control input
    return x, A


def static_figure(x) -> None:
    plt.rcParams["font.size"] = 14
    fig1 = plt.figure(figsize=(8, 8), tight_layout=True)
    ax1 = fig1.add_subplot(211)
    for i in range(x.shape[0]):
        ax1.plot(x[i, 0, :], label='Agent {}'.format(i))
    ax1.set_title("Agent state for x-axis")
    ax1.set_xlim(0, x.shape[2]-1)
    
    ax2 = fig1.add_subplot(212)
    for i in range(x.shape[0]):
        ax2.plot(x[i, 1, :], label='Agent {}'.format(i))
    ax2.set_title("Agent state for y-axis")
    ax2.set_xlim(0, x.shape[2]-1)
    
    fig1.savefig('Result.png')


def RenderGIF(x, A, fps=30, interval=10) -> None:
    plt.rcParams["font.size"] = 14
    fig2 = plt.figure(figsize=(8, 8), tight_layout=True)
    ax3 = fig2.add_subplot(111)
    ax3.set_xlim(min(x[:, 0, 0]) - 0.1, max(x[:, 0, 0]) + 0.1)
    ax3.set_ylim(min(x[:, 1, 0]) - 0.1, max(x[:, 1, 0]) + 0.1)
    ax3.set_title("Agent Trajectory")
    scatter_plots = [ax3.scatter(x[I, 0, 0], x[I, 1, 0], marker='o', s=100) for I in range(x.shape[0])]
    lines = []

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] > 0:
                line, = ax3.plot([x[i, 0, 0], x[j, 0, 0]], [x[i, 1, 0], x[j, 1, 0]], 'k-', lw=1)
                lines.append(line)

    def update(frame):
        for I in range(x.shape[0]):
            scatter_plots[I].set_offsets(np.column_stack([x[I, 0, frame], x[I, 1, frame]]))
        for line, (i, j) in zip(lines, np.transpose(np.nonzero(A))):
            line.set_data([x[i, 0, frame], x[j, 0, frame]], [x[i, 1, frame], x[j, 1, frame]])
        return scatter_plots + lines

    anim = animation.FuncAnimation(fig2, update, frames=x.shape[2], interval=interval, blit=True)
    anim.save('Result.gif', writer='pillow', fps=fps)


if __name__ == '__main__':
    state, adj_matrix = Consensus(2, 301)
    RenderGIF(state, adj_matrix)
    static_figure(state)