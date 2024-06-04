import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

np.random.seed(1) # Set random seed


class network():
    def __init__(self):
        self.L = np.empty

    def Laplacian(self) -> np.ndarray:
        # Set graph Laplacian matrix
        # If you want to change network, please edit the below matrix 'a'
        a = np.array([[0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                      [1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1],
                      [1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1],
                      [1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0],
                      [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                      [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1],
                      [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                      [0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1],
                      [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1],
                      [0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                      [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
                      [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
                      [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
                      [0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
                      [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
                      [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0]])
        A = a + a.T
        D = np.diag(np.sum(A, axis=0))
        L = D - A

        return L


def Consensus(dimention: int, step: int) -> np.ndarray:
    G = network()
    L = G.Laplacian()
    n = L.shape[0] # Get the number of nodes from matrix shape
    x = np.zeros((n, dimention, step))
    x[:, :, 0] = np.random.uniform(size=(n, dimention)) # Generate the state vector with random function
    for k in range(1, step):
        x[:, :, k] = x[:, :, k - 1] - 0.005 *  L @ x[:, :, k - 1] # Calculate control input
    return x


def static_figure(x) -> None:
    plt.rcParams["font.size"] = 14
    fig1 = plt.figure(figsize=(8, 6), tight_layout=True)
    ax1 = fig1.add_subplot(211)
    for i in range(x.shape[0]):
        ax1.plot(x[i, 0, :], marker='o', markersize=1, label='Agent {}'.format(i))
    ax1.set_title("Agent state for x-axis")
    ax1.set_xlim(0, x.shape[2]-1)
    
    ax2 = fig1.add_subplot(212)
    for i in range(x.shape[0]):
        ax2.plot(x[i, 1, :], marker='o', markersize=1, label='Agent {}'.format(i))
    ax2.set_title("Agent state for y-axis")
    ax2.set_xlim(0, x.shape[2]-1)
    
    fig1.savefig('Result.png')


def RenderGIF(x, fps=40, interval=10) -> None:
    plt.rcParams["font.size"] = 14
    fig2 = plt.figure(figsize=(8, 8), tight_layout=True)
    ax3 = fig2.add_subplot(111)
    ax3.set_xlim(min(x[:, 0, 0]) - 0.1, max(x[:, 0, 0]) + 0.1)
    ax3.set_ylim(min(x[:, 1, 0]) - 0.1, max(x[:, 1, 0]) + 0.1)
    ax3.set_title("Agent Trajectory")
    scatter_plots = [ax3.scatter(x[I, 0, 0], x[I, 1, 0], marker='o', s=100) for I in range(x.shape[0])] 

    def update(frame):
        for I in range(x.shape[0]):
            scatter_plots[I].set_offsets(np.column_stack([x[I, 0, frame], x[I, 1, frame]]))
        return scatter_plots

    anim = animation.FuncAnimation(fig2, update, frames=x.shape[2], interval=interval, blit=True)
    anim.save('Result.gif', writer='pillow', fps=fps)


if __name__ == '__main__':
    state = Consensus(2, 121)
    RenderGIF(state)
    static_figure(state)
