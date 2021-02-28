clear all; clc; close all;
%% Select name
sp = "circle"; % "circle" "star" "torus" "maze" "separation" "dumbbell"
%% Select files number
itern=[1 100 180 240];
% circle: [1 100 180 240] % star: [1 20 150 260] % torus: [1 150 300 460]
% maze:[1 20 50 80] % separation: [1 20 30 100] % dumbbell: [1 50 150 300]
%% Load data
for i=itern
    GPUdata{i} = readNPY(sprintf('./2d_gpu/%s_%d.npy',sp,i));
end
%% Domain and time step
if sp=="dumbbell"
    Nx=400; Ny=200; h=1/Nx;
    x=linspace(0,2,Nx); y=linspace(0,1,Ny);
elseif sp=="maze"
    Nx=100; Ny=100; h=1/Nx;
    x=linspace(0,1,Nx); y=linspace(0,1,Ny);
else
    Nx=200; Ny=200; h=1/Nx;
    x=linspace(0,1,Nx); y=linspace(0,1,Ny);
end
dt=0.1*h^2;
%% Show figures
for i=itern
    figure(i); clf;
    [c,H]= contourf(x,y,GPUdata{i}',[0 0],'facecolor',[115 82 68]./255);
    drawnow
    axis([x(1) x(end) y(1) y(end)])
    axis image
    set(gca,'fontsize',15)
end
