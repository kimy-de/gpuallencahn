# Libraries
import numpy as np
import initial2d
import argparse
import python_cpu
import pytorch_code
import matplotlib.pyplot as plt
import graph

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Allen-Cahn Equation in 2D')
    parser.add_argument('--maxit', default=2001, type=int, help='number of iterations')
    parser.add_argument('--nx', default=100, type=int, help='Nx')
    parser.add_argument('--ny', default=100, type=int, help='Ny')
    parser.add_argument('--init', default='dumbell', type=str, help='initial condition') # ['circle', 'dumbell', 'star', 'separation', 'torus', 'maze']
    parser.add_argument('--mode', default=1, type=int, help='code') # 0: pytorch gpu, 1: pytorch cpu, 2: python cpu
    parser.add_argument('--save', default=0, type=int, help='npy files') # 0: No, 1: Yes
    parser.add_argument('--l2', default=0, type=int, help='l2 loss') # 0: No, 1: Yes
    args = parser.parse_args()
    print(args)

    # Parameter setting
    h = 1/args.nx
    h2 = h**2
    dt = .1*h**2
    eps = 10 * h / (2 * np.sqrt(2) * np.arctanh(0.9))
    maxtime = dt * args.maxit
    pnp = np.zeros([args.nx + 2, args.ny + 2])
    # Initial condition
    x, y, pn = initial2d.initial_value(args.nx, args.ny, eps, args.init)

    # Numerical solution
    if (args.mode != 2) or (args.l2 == 1):
        if args.l2 == 0:
            mode = args.mode
        else:
            mode = 0
        pnusols, gputime = pytorch_code.fdm_pytorch(args.nx, args.ny, pn, h2, dt, eps,
                                                    args.maxit, args.init, args.save, mode)
        graph.result(pnusols, x, y, maxtime, args.mode, args.init)

    if (args.mode == 2) or (args.l2 == 1):
        nusols, cputime = python_cpu.fdm(args.nx, args.ny, h2, dt, eps, pn, pnp, args.maxit, args.init, args.save)
        graph.result(nusols, x, y, maxtime, args.mode, args.init)

    if args.l2 == 1:
        error = []
        for i in range(len(nusols)):
            l2 = np.sqrt(np.sum((pnusols[i] - nusols[i]) ** 2) / (args.nx * args.ny))
            error.append(l2)

        plt.plot(error)
        plt.title("L2 error vs time")
        plt.savefig("./data/error.png")









# Boundary Condition
pn[0, :] = np.copy(pn[1, :])
pn[-1, :] = np.copy(pn[-2, :])
pn[:, 0] = np.copy(pn[:, 1])
pn[:, -1] = np.copy(pn[:, -2])

# Explicit Allen-Cahn
coef1 = dt / (eps ** 2)
coef2 = dt / h2
part1 = (1 + coef1) * pn - coef1 * pn ** 3

for i in range(1, Nx + 1):
    for j in range(1, Ny + 1):
        pnp[i, j] = coef2 * (pn[i - 1, j] + pn[i + 1, j] + pn[i, j - 1]
                    + pn[i, j + 1] - 4.0 * pn[i, j])

pn = part1 + pnp







