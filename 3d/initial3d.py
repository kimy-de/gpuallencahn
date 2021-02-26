import numpy as np

def initial_value(Nx, Ny, Nz, eps, init):
    pn = np.zeros([Nx + 2, Ny + 2, Nz + 2])
    # Setting Parameters
    h = 1 / Nx

    if init == 'dumbell':
        x = np.linspace(-0.5 * h, 2 + 0.5 * h, Nx + 2)
        y = np.linspace(-0.5 * h, 1 + 0.5 * h, Ny + 2)
        z = np.linspace(-0.5 * h, 1 + 0.5 * h, Nz + 2)

    elif (init == 'star') or (init == 'maze') or (init == 'torus'):
        x = np.linspace(-1 - 0.5 * h, 1 + 0.5 * h, Nx + 2)
        y = np.linspace(-1 - 0.5 * h, 1 + 0.5 * h, Ny + 2)
        z = np.linspace(-1 - 0.5 * h, 1 + 0.5 * h, Nz + 2)

    else:
        x = np.linspace(-0.5 * h, h * (Nx + 0.5), Nx + 2)
        y = np.linspace(-0.5 * h, h * (Ny + 0.5), Ny + 2)
        z = np.linspace(-0.5 * h, h * (Nz + 0.5), Nz + 2)

    if init == 'separation':
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    pn[i, j, k] = 0.1 * (2 * np.random.rand() - 1)

    elif init == 'sphere':
        R0 = 0.25
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    pn[i, j, k] = np.tanh(
                        (R0 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2 + (z[k] - 0.5) ** 2)) / (np.sqrt(2) * eps))

    elif init == 'dumbell':
        R0 = 0.25
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    if x[i] > 0.4 and x[i] < 1.6 and y[j] > 0.4 and y[j] < 0.6 and z[k] > 0.4 and z[k] < 0.6:
                        pn[i, j, k] = 1.0
                    else:
                        pn[i, j, k] = np.tanh(
                            (R0 - np.sqrt((x[i] - 0.3) ** 2 + (y[j] - 0.5) ** 2 + (z[k] - 0.5) ** 2)) / (
                                        np.sqrt(2) * eps)) + np.tanh(
                            (R0 - np.sqrt((x[i] - 1.7) ** 2 + (y[j] - 0.5) ** 2 + (z[k] - 0.5) ** 2)) / (
                                        np.sqrt(2) * eps)) + 1

    elif init == 'star':
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    theta = np.arctan2(z[k], x[i])
                    pn[i, j, k] = np.tanh(
                        (0.7 + 0.2 * np.cos(6 * theta) - (np.sqrt(x[i] ** 2 + 2 * y[j] ** 2 + z[k] ** 2))) / (
                                    np.sqrt(2.0) * eps))


    elif init == 'torus':
        r1 = 0.6
        r2 = 0.3

        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    pn[i, j, k] = np.sqrt(z[k] ** 2 + (np.sqrt(x[i] ** 2 + y[j] ** 2) - r1) ** 2) - r2

    elif init == 'maze':
        a1 = 9
        b1 = 9
        a2 = a1
        b2 = b1
        a3 = 3
        b3 = 3
        for i in range(Nx + 2):
            for j in range(Ny + 2):
                for k in range(Nz + 2):
                    pn[i, j, k] = -1.0
                    if ((k >= 1 + 1.2 * b3 and k <= Nz - 1.2 * b3) and (
                            (i > a1 and i < a1 + 1.5 * a2 and j > b1 and j < Ny - 0.5 * b2) or
                            (i > a1 and i < Nx - a1 and j > b1 and j < b1 + 1.5 * b2) or
                            (i > Nx - a1 - 1.5 * a2 and i < Nx - a1 and j > b1 and j < Ny - 2 * b2 - b3) or
                            (i > a1 + 2 * a2 + 2 * a3 and i < Nx - a1 and j > Ny - 3.5 * b2 - b3 and j < Ny - 2 * b2 - b3) or
                            (i > a1 + 2 * a2 + 2 * a3 and i < 2 * a1 + 2.5 * a2 + 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < Ny - 2 * b2 - b3) or
                            (i > a1 + 2 * a2 + 2 * a3 and i < Nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < b1 + 3.5 * b2 + 2 * b3) or
                            (i > Nx - a1 - 3.5 * a2 - 2 * a3 and i < Nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < Ny - 4 * b2 - 3 * b3))):
                        pn[i, j, k] = 1.0

    return pn