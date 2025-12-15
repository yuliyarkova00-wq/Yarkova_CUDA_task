#pragma once
#include <cuda_runtime.h>


__global__ void it0_kernel(
    double* u0,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double* err
);


void it0_kernel_launcher(
    double* u0,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double& err
);

__global__ void it1_kernel(
    double *u0,
    double *u1,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double *err
);

void it1_kernel_launcher(
    double *u0,
    double *u1,
    int nx, int ny, int nz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau,
    double &err
);

__global__ void main_kernel(
    double *u0,
    double *u1,
    double *u2,
    int nx, int ny, int nz,
    int gx, int gy, int gz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau, double t,
    double *err
);

void main_kernel_launcher(
    double *u0,
    double *u1,
    double *u2,
    int nx, int ny, int nz,
    int gx, int gy, int gz,
    int sx, int sy, int sz,
    double dx, double dy, double dz,
    double a, double tau, double t,
    double &err 
);
