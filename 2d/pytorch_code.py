import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class Net(nn.Module):
    def __init__(self, h2, dt, eps, device):
        super(Net, self).__init__()
        # 2nd order differencing filter
        self.delta = torch.Tensor([[[[0., 1., 0.], [1., -4., 1], [0., 1., 0.]]]]).to(device)
        self.pad = nn.ReplicationPad2d(1) # Replication pad for boundary condition
        self.alpha = dt / eps ** 2
        self.beta = dt / h2

    def forward(self, x):
        u_pad = self.pad(x) # boundary condition
        reaction = F.conv2d(u_pad, self.delta) # reaction term
        x = (1 + self.alpha) * x - self.alpha * x ** 3 + self.beta * reaction

        return x

def fdm_pytorch(nx, ny, init_arr, h2, dt, eps, maxit, init, save, mode):

    if mode == 0:
        device = "cuda:0"
        print("GPU version")
    else:
        device = "cpu"
        print("CPU version")

    model = Net(h2, dt, eps, device).to(device)

    # Initial value
    img = torch.FloatTensor(init_arr[1:-1, 1:-1]).view(-1, 1, nx, ny).to(device)

    start = time.time()
    pnusols = []
    number = 0

    with torch.no_grad():

        for step in range(maxit):
            u = model(img)
            img = u # phi^(n+1) <- f(phi^n)
            pnusols.append(img.view(nx, ny).cpu().numpy())

            if (save == 1) and (mode == 0):
                if step % 50 == 0:
                    with open('./data/2d_gpu/' + init +'_' + str(number) + '.npy', 'wb') as f:
                        np.save(f, img.view(nx, ny).cpu().numpy())
                    f.close()
                    number += 1

    runtime = time.time() - start
    print("Pytorch Runtime: ", runtime)

    return np.array(pnusols), runtime