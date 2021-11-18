# Fast and Accurate Numerical Solution of Allen–Cahn Equation
#### Yongho Kim, Gilnam Ryu, and  Yongho Choi (2021) Fast and Accurate Numerical Solution of Allen–Cahn Equation, Mathematical Problems in Engineering, accepted
Simulation speed depends on code structures, hence it is crucial how to build a fast algorithm. We solve the Allen–Cahn equation by an explicit finite difference method, so it requires grid calculations implemented by many for-loops in the simulation code. In terms of programming, many for-loops make the simulation speed slow. We propose a model architecture containing a pad and a convolution operation on the Allen–Cahn equation for fast computations while maintaining accuracy. Also, the GPU operation is used to boost up the speed more. In this way, the simulation of other differential equations can be improved. In this paper, various numerical simulations are conducted to confirm that the Allen–Cahn equation follows motion by mean curvature and phase separation in two-dimensional and three-dimensional spaces. Finally, we demonstrate that our algorithm is much faster than an unoptimized code and the CPU operation.


## Allen-Cahn Equation
<p align="center">
<img width="509" alt="model_f" src="https://user-images.githubusercontent.com/52735725/119031362-4e5a0280-b9ab-11eb-8576-07262c00eb3d.png">
  <img width="708" alt="스크린샷 2021-05-20 20 43 03" src="https://user-images.githubusercontent.com/52735725/119032019-038cba80-b9ac-11eb-9c79-c94fb79ec825.png">

</p>



## Implementations

```python
"""
Number of iterations: --maxit
Grid size: --Nx, --Ny, (and --Nz)
Initial condition: --init 
# 2D: ['circle', 'dumbbell', 'star', 'separation', 'torus', 'maze'] 
# 3D: ['sphere', 'dumbbell', 'star', 'separation', 'torus', 'maze']
Operation mode: --mode # 0: pytorch gpu, 1: pytorch cpu, 2: python cpu
Saving npy files: --save  # 0: No, 1: Yes
"""
```
```
python 2d.py --mode 0 --init star
```
```
python 2d.py --mode 0 --init star --maxi 2001 --save 1
```
```
python 3d.py --mode 0 --init maze
```

## Results
<p align="center">
<img width="786" alt="Screen Shot 2021-04-10 at 8 58 35 AM" src="https://user-images.githubusercontent.com/52735725/119032113-199a7b00-b9ac-11eb-8b9d-b49b00c6bde9.png">
  <img width="426" alt="스크린샷 2021-04-29 22 00 30" src="https://user-images.githubusercontent.com/52735725/119032166-2ae38780-b9ac-11eb-9e0c-096ee1f467f2.png">

</p>
