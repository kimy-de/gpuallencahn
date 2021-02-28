clear all; clc; close all;
%% Select name
sp = "sphere"; % "sphere" "star" "torus" "maze" "separation" "dumbbell"
%% Select files number
itern=[1 10 20 28];
% sphere: [1 10 20 28]  % star: [1 5 15 40]  % torus: [1 10 16 20]
% maze: [1 10 25 35]  % separation: [1 5 10 20]  % dumbbell: [1 5 10 20]
%% Load data
for i=itern
    GPUdata{i} = readNPY(sprintf('./3d_gpu/%s_%d.npy',sp,i));
end
%% Domain and time step
if sp=="star" || sp=="torus" || sp=="maze"
    Nx=100; Ny=100; Nz=100; h=1/Nx;
    x=linspace(-1,1,Nx); y=linspace(-1,1,Ny); z=linspace(-1,1,Nz);
    
elseif sp=="separation" || sp=="sphere"
    Nx=100; Ny=100; Nz=100; h=1/Nx;
    x=linspace(0,1,Nx); y=linspace(0,1,Ny); z=linspace(0,1,Nz);
    
elseif sp=="dumbbell"
    Nx=200; Ny=100; Nz=100; h=1/Nx;
    x=linspace(0,2,Nx); y=linspace(0,1,Ny); z=linspace(0,1,Nz);
end
dt=0.1*h^2;
%% Generate meshgrid
[x y z] = meshgrid(y,x,z);
%% Show figures
for i=itern
    figure(i); clf; fv2 = isosurface(x,y,z,GPUdata{i},0);
    c=fv2.vertices; pp = patch(fv2); pp.FaceColor = [194 150 130]./255;
    pp.EdgeColor = 'none'; daspect([1 1 1]);
    view(-50,13); camlight;lighting gouraud; grid on; drawnow
    axis([x(1) x(end) y(1) y(end) z(1) z(end)])
    set(gca,'fontsize',15)
end