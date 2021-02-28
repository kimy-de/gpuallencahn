import numpy as np

# separation 3.007189228024808e-06
# sphere 1.1144527752384542e-06
# torus 1.9067423619924316e-06
# star 1.5604937140470243e-06
# dumbbell 1.1984330058904703e-06
# maze 3.363573269338306e-06

ini = 'star'
n = 41

pathcpu = './data/3d_cpu/'
pathgpu = './data/3d_gpu/'

rmse_list = []

for i in range(n):

     with open(pathcpu+ini+'_'+str(i)+'.npy', 'rb') as f:
         a = np.load(f)[1:-1,1:-1,1:-1]
     f.close()
     with open(pathgpu+ini+'_'+str(i)+'.npy', 'rb') as f:
         b = np.load(f)
     f.close()
     rmse = np.sqrt(np.mean((a-b)**2))
     rmse_list.append(rmse)

print(sum(rmse_list)/len(rmse_list))
