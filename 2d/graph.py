import matplotlib.pyplot as plt

def result(nusols, x, y, maxtime, mode, init):

    if mode == 0:
        name = "pytorch_gpu"
    elif mode == 1:
        name = "pytorch_cpu"
    elif mode == 2:
        name = "python_cpu"
    else:
        print("Error: Can't search the mode.")
        exit(0)

    # Plot
    n = len(nusols)
    x = x[1:-1].reshape(-1, 1)
    y = y[1:-1].reshape(-1, 1)
    extend = [x.min(), x.max(), y.min(), y.max()]
    plt.figure(figsize=(10, 5))
    plt.subplot(231)
    plt.imshow(nusols[0], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    #plt.xlabel('x', fontsize=15)
    #plt.ylabel('y', fontsize=15)
    plt.title("t=0", fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(nusols[int(.2 * n)], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    plt.title("t=" + str(round(maxtime * .2, 4)), fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(nusols[int(.4 * n)], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    plt.title("t=" + str(round(maxtime * .4, 4)), fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(nusols[int(.6 * n)], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    plt.title("t=" + str(round(maxtime * .6, 4)), fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(nusols[int(.8 * n)], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    plt.title("t=" + str(round(maxtime * .8, 4)), fontsize=20)
    plt.clim(-1, 1)
    plt.axis('off')

    plt.subplot(236)
    plt.imshow(nusols[-1], interpolation='nearest', cmap='jet',
               extent=extend, origin='lower', aspect='auto')
    plt.title("t=" + str(round(maxtime, 4)), fontsize=20)
    plt.axis('off')
    plt.clim(-1, 1)
    plt.savefig("./data/" + name + "_" + init + ".png")
    plt.close()