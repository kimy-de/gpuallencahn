# Libraries
import numpy as np
import initial3d
import argparse
import python_cpu
import pytorch_code
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Allen-Cahn Equation in 3D')
    parser.add_argument('--maxit', default=501, type=int, help='number of iterations')
    parser.add_argument('--nx', default=100, type=int, help='Nx')
    parser.add_argument('--ny', default=100, type=int, help='Ny')
    parser.add_argument('--nz', default=100, type=int, help='Nx')
    parser.add_argument('--init', default='sphere', type=str, help='initial condition') # ['sphere', 'dumbbell', 'star', 'separation', 'torus', 'maze']
    parser.add_argument('--mode', default=1, type=int, help='code') # 0: pytorch gpu, 1: pytorch cpu, 2: python cpu
    parser.add_argument('--save', default=0, type=int, help='npy files') # 0: No, 1: Yes
    parser.add_argument('--l2', default=0, type=int, help='l2 loss') # 0: No, 1: Yes
    args = parser.parse_args()
    print(args)

    # Parameter setting
    h = 1/args.nx
    h2 = h**2
    dt = .1*h**2
    eps = 12 * h / (2 * np.sqrt(2) * np.arctanh(0.9))
    maxtime = dt * args.maxit
    pnp = np.zeros([args.nx + 2, args.ny + 2, args.nz + 2])

    # Initial condition
    pn = initial3d.initial_value(args.nx, args.ny, args.nz,  eps, args.init)

    # Numerical solution
    if (args.mode != 2) or (args.l2 == 1):
        if args.l2 == 0:
            mode = args.mode
        else:
            mode = 0
        pnusols, gputime = pytorch_code.fdm_pytorch(args.nx, args.ny, args.nz, pn, h2, dt, eps,
                                                    args.maxit, args.init, args.save, mode)

    if (args.mode == 2) or (args.l2 == 1):
        nusols, cputime = python_cpu.fdm(args.nx, args.ny, args.nz, h2, dt, eps, pn, pnp, args.maxit, args.init, args.save)

    if args.l2 == 1:
        error = []
        for i in range(len(nusols)):
            l2 = np.sqrt(np.sum((pnusols[i] - nusols[i]) ** 2) / (args.nx * args.ny * args.nz))
            error.append(l2)

        plt.plot(error)
        plt.title("L2 error vs time")
        plt.savefig("./data/error.png")
















