# Initial Values
import numpy as np
import matplotlib.pyplot as plt


def initial_value(Nx, Ny, eps, initial_condition):
    pn = np.zeros([Nx + 2, Ny + 2])
    # Setting Parameters
    h = 1 / Nx

    if initial_condition == 'dumbell':
        x = np.linspace(-0.5 * h, 2 + h * 0.5, Nx + 2)
        y = np.linspace(-0.5 * h, 1 + h * 0.5, Ny + 2)

    else:
        x = np.linspace(-0.5 * h, h * (Nx + 0.5), Nx + 2)
        y = np.linspace(-0.5 * h, h * (Ny + 0.5), Ny + 2)

    if initial_condition == 'separation':
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                pn[i, j] = 0.1 * (2 * np.random.rand() - 1)

    elif initial_condition == 'circle':
        R0 = 0.25
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                pn[i, j] = np.tanh((R0 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))

    elif initial_condition == 'dumbell':
        R0 = 0.2
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                if ((x[i] > 0.4 and x[i] < 1.6) and (y[j] > 0.4 and y[j] < 0.6)):
                    pn[i, j] = 1.0
                else:
                    pn[i, j] = (1 + np.tanh(
                        (R0 - np.sqrt((x[i] - 0.3) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps)) +
                        np.tanh((R0 - np.sqrt((x[i] - 1.7) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps)))

    elif initial_condition == 'star':
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                if x[i] > 0.5:
                    theta = np.arctan2(y[j] - 0.5, x[i] - 0.5)
                    pn[i, j] = np.tanh(
                        (0.25 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) /
                        (np.sqrt(2.0) * eps))
                else:
                    theta = np.pi + np.arctan2(y[j] - 0.5, x[i] - 0.5)
                    pn[i, j] = np.tanh(
                        (0.25 + 0.1 * np.cos(6 * theta) - (np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2))) /
                        (np.sqrt(2.0) * eps))

    elif initial_condition == 'torus':
        r1 = 0.4
        r2 = 0.3

        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                pn[i, j] = (np.tanh(
                    (r1 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps)) -
                    np.tanh((r2 - np.sqrt((x[i] - 0.5) ** 2 + (y[j] - 0.5) ** 2)) / (np.sqrt(2) * eps))) - 1

    elif initial_condition == 'maze':
        a1 = 8
        b1 = 8
        a2 = a1
        b2 = b1
        a3 = 3
        b3 = 3
        for i in range(1, Nx + 1):
            for j in range(1, Ny + 1):
                pn[i, j] = -1.0
                if ((i > a1 and i < a1 + 1.5 * a2 and j > b1 and j < Ny - 0.5 * b2) or
                        (i > a1 and i < Nx - a1 and j > b1 and j < b1 + 1.5 * b2) or
                        (i > Nx - a1 - 1.5 * a2 and i < Nx - a1 and j > b1 and j < Ny - 2 * b2 - b3) or
                        (i > a1 + 2 * a2 + 2 * a3 and i < Nx - a1 and j > Ny - 3.5 * b2 - b3 and j < Ny - 2 * b2 - b3) or
                        (i > a1 + 2 * a2 + 2 * a3 and i < 2 * a1 + 2.5 * a2 + 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < Ny - 2 * b2 - b3) or
                        (i > a1 + 2 * a2 + 2 * a3 and i < Nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < b1 + 3.5 * b2 + 2 * b3) or
                        (i > Nx - a1 - 3.5 * a2 - 2 * a3 and i < Nx - a1 - 2 * a2 - 2 * a3 and j > b1 + 2 * b2 + 2 * b3 and j < Ny - 4 * b2 - 3 * b3)):
                    pn[i, j] = 1.0

    else:
        print("Error: Can't search the condition.")
        exit(0)

    # Graph of Initial Value
    x = x[1:-1]
    y = y[1:-1]
    tpn = pn[1:-1, 1:-1]
    if initial_condition == 'dumbell':
        plt.figure(figsize=(4, 2))
    else:
        plt.figure(figsize=(4, 4))
    plt.imshow(tpn, interpolation='nearest', cmap='jet',
               extent=[x.min(), x.max(), y.min(), y.max()],
               origin='lower', aspect='auto')

    plt.title('$u(x,y,t=0)$', fontsize=20)
    plt.clim(-1, 1)
    plt.savefig("./data/" + initial_condition + ".png")

    return x, y, pn
