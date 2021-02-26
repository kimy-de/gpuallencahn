import time
import numpy as np

# Numerical Simulation on CPU(Numpy code)
def fdm(Nx, Ny, Nz, h2, dt, eps, pn, pnp, maxit, init, save):

    nusols = []
    start = time.time()
    number = 0

    for step in range(maxit):

        # Boundary Condition
        pn[0, :, :] = np.copy(pn[1, :, :])
        pn[-1, :, :] = np.copy(pn[-2, :, :])
        pn[:, 0, :] = np.copy(pn[:, 1, :])
        pn[:, -1, :] = np.copy(pn[:, -2, :])
        pn[:, :, 0] = np.copy(pn[:, :, 1])
        pn[:, :, -1] = np.copy(pn[:, :, -2])

        # Explicit Allen-Cahn
        coef1 = dt / (eps ** 2)
        coef2 = dt / h2
        part1 = (1 + coef1) * pn - coef1 * pn ** 3
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                for k in range(1, Nz + 1):
                    pnp[i, j, k] = coef2 * (pn[i - 1, j, k] + pn[i + 1, j, k] + pn[i, j - 1, k] + pn[i, j + 1, k] + pn[
                        i, j, k + 1] + pn[i, j, k - 1] - 6.0 * pn[i, j, k])

        pn = part1 + pnp

        nusols.append(pn[1:-1, 1:-1, 1:-1])
        if save == 1:
            if step % 50 == 0:
                with open('./data/3d_cpu/' + init + '_' + str(number) + '.npy', 'wb') as f:
                    np.save(f, pn[1:-1, 1:-1, 1:-1])
                number += 1
                f.close()

    cputime = time.time() - start
    print("Python Runtime: ", cputime)

    return np.array(nusols), cputime