import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np


class Net(nn.Module):
    def __init__(self, h2, dt, eps, device):
        super(Net, self).__init__()

        self.h2 = h2
        self.dt = dt
        self.eps = eps
        self.delta = torch.Tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
                                   [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]).to(device)
        self.delta = torch.unsqueeze(self.delta, 0)
        self.delta = torch.unsqueeze(self.delta, 0)
        self.pad = nn.ReplicationPad3d(1)
        self.alpha = self.dt / eps ** 2

    def forward(self, x):
        u_pad = self.pad(x)
        dff = F.conv3d(u_pad, self.delta)
        x = (1 + self.alpha) * x - self.alpha * x ** 3 + self.dt * dff / self.h2

        return x

def fdm_pytorch(nx, ny, nz,  init_arr, h2, dt, eps, maxit, init, save, mode):

    if mode == 0:
        device = "cuda:0"
        print("GPU version")
    else:
        device = "cpu"
        print("CPU version")

    model = Net(h2, dt, eps, device).to(device)

    # Initial value
    img = torch.FloatTensor(init_arr[1:-1, 1:-1, 1:-1]).view(-1, 1, nx, ny, nz).to(device)

    start = time.time()
    pnusols = []
    number = 0

    with torch.no_grad():

        for step in range(maxit):
            u = model(img)
            img = u
            pnusols.append(img.view(nx, ny, nz).cpu().numpy())

            if (save == 1) and (mode == 0):
                if step % 50 == 0:
                    with open('./data/3d_gpu/' + init +'_' + str(number) + '.npy', 'wb') as f:
                        np.save(f, img.view(nx, ny, nz).cpu().numpy())
                    f.close()
                    number += 1

    runtime = time.time() - start
    print("Pytorch Runtime: ", runtime)

    return np.array(pnusols), runtime