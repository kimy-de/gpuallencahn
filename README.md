# GPU Accelerated Numerical Simulation of Allen-Cahn Equation
abstract

## Allen-Cahn Equation
수식, 결과 그림

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
속도 테이블

## Citation
ㅍㅍ
