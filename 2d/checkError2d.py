import numpy as np

# 2D  RMSE
# separation 1.7530007240182663e-06
# circle 5.510502296905582e-07
# torus 7.521386005333273e-07
# star 6.550860553223637e-07
# dumbbell 7.02910924592743e-07
# maze 2.219102476771937e-07

ini = 'circle'
n = 241
pathcpu = './data/2d_cpu/'
pathgpu = './data/2d_gpu/'

rmse_list = []

for i in range(n):

     with open(pathcpu+ini+'_'+str(i)+'.npy', 'rb') as f:
         a = np.load(f)[1:-1,1:-1]
     f.close()
     with open(pathgpu+ini+'_'+str(i)+'.npy', 'rb') as f:
         b = np.load(f)
     f.close()
     rmse = np.sqrt(np.mean((a-b)**2))
     rmse_list.append(rmse)

print(sum(rmse_list)/len(rmse_list))
