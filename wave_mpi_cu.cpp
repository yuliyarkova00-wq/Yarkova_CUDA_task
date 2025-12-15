#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include "kernel.cuh"
#include <cuda_runtime.h>

const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;


double an_sol(double x, double y, double z, double t) {
    const double at = 0.5 * sqrt(4.0 / (Lx*Lx) + 1.0/(Ly*Ly) + 1.0/(Lz*Lz));
    const double pi = M_PI;
    return sin(2.0 * pi * x / Lx) * sin(pi * y / Ly) * sin(pi * z / Lz) * cos(at * t + 2.0 * pi);
}


double compute_p(double x, double y, double z) {
    return an_sol(x, y, z, 0.0);
}


void block_decompose(int N, int procs, int rank_coord, int &local_n, int &start_idx) {
    int base = N / procs;
    int rem  = N % procs;
    if (rank_coord < rem) {
        local_n = base + 1;
        start_idx = rank_coord * local_n;
    } else {
        local_n = base;
        start_idx = rem * (base + 1) + (rank_coord - rem) * base;
    }
}


inline size_t idx(int i, int j, int k, int nx, int ny, int nz) {
    return (size_t)i * ny * nz + (size_t)j * nz + (size_t)k;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    int n_devices_available = 0;
    cudaGetDeviceCount(&n_devices_available);
    int num_d = world_rank % n_devices_available;
    cudaSetDevice(num_d);
    int Nx = 256;
    int Ny = 256;
    int Nz = 256;
    double tau = 0.002;
    int Nt = 50;
    double a = 1 / (4 * M_PI * M_PI);
    // double a = (Lx*Lx*Ly*Ly*Lz*Lz)/(Lx*Lx*Lz*Lz + Ly*Ly*Lz*Lz + Lx*Lx*Ly*Ly);

    if (argc >= 5) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
        Nt = atoi(argv[4]);
    }
    if (argc >= 6) tau = atof(argv[5]);

   
    int dims[3] = {0,0,0};
    MPI_Dims_create(world_size, 3, dims); 
    int periods[3] = {0,0,0}; 
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);

    int coords[3];
    MPI_Cart_coords(cart_comm, world_rank, 3, coords);

    
    int nbr_left, nbr_right, nbr_down, nbr_up, nbr_back, nbr_front;
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_left, &nbr_right);   
    MPI_Cart_shift(cart_comm, 1, 1, &nbr_down, &nbr_up);       
    MPI_Cart_shift(cart_comm, 2, 1, &nbr_back, &nbr_front);    

   
    int local_nx, local_ny, local_nz;
    int start_x, start_y, start_z;
    block_decompose(Nx, dims[0], coords[0], local_nx, start_x);
    block_decompose(Ny, dims[1], coords[1], local_ny, start_y);
    block_decompose(Nz, dims[2], coords[2], local_nz, start_z);

   
    int gx = local_nx + 2;
    int gy = local_ny + 2;
    int gz = local_nz + 2;
    size_t total = (size_t)gx * gy * gz;

    std::vector<double> u0(total, 0.0), u1(total, 0.0), u2(total, 0.0);
    double *dev_u0 = nullptr;
    double *dev_u1 = nullptr;
    double *dev_u = nullptr;
    cudaMalloc(&dev_u0, total*sizeof(double));
    double dx = Lx / (Nx - 1);
    double dy = Ly / (Ny - 1);
    double dz = Lz / (Nz - 1);
   
    
    double t_beg = MPI_Wtime();

    double local_max_err0 = 0.0;
    
    it0_kernel_launcher(
        dev_u0,
        local_nx, local_ny, local_nz,
        start_x, start_y, start_z,
        dx, dy, dz,
        local_max_err0
    );
    cudaMemcpy(u0.data(), dev_u0, total*sizeof(double), cudaMemcpyDeviceToHost);
    if (nbr_left == MPI_PROC_NULL) {
        int gi = 0;
        for (int j = 1; j <= local_ny; ++j)
            for (int k = 1; k <= local_nz; ++k) {
                int glob_i = start_x - 1; 
                int glob_i_use = std::max(0, start_x - 1);
                double x = glob_i_use * dx;
                double y = (start_y + (j-1)) * dy;
                double z = (start_z + (k-1)) * dz;
                u0[idx(gi,j,k,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }
    if (nbr_right == MPI_PROC_NULL) {
        int gi = gx-1;
        for (int j = 1; j <= local_ny; ++j)
            for (int k = 1; k <= local_nz; ++k) {
                int glob_i_use = std::min(Nx-1, start_x + (local_nx));
                double x = glob_i_use * dx;
                double y = (start_y + (j-1)) * dy;
                double z = (start_z + (k-1)) * dz;
                u0[idx(gi,j,k,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }
  
    if (nbr_down == MPI_PROC_NULL) {
        int gj = 0;
        for (int i = 1; i <= local_nx; ++i)
            for (int k = 1; k <= local_nz; ++k) {
                int glob_j_use = std::max(0, start_y - 1);
                double x = (start_x + (i-1)) * dx;
                double y = glob_j_use * dy;
                double z = (start_z + (k-1)) * dz;
                u0[idx(i,gj,k,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }
    if (nbr_up == MPI_PROC_NULL) {
        int gj = gy-1;
        for (int i = 1; i <= local_nx; ++i)
            for (int k = 1; k <= local_nz; ++k) {
                int glob_j_use = std::min(Ny-1, start_y + (local_ny));
                double x = (start_x + (i-1)) * dx;
                double y = glob_j_use * dy;
                double z = (start_z + (k-1)) * dz;
                u0[idx(i,gj,k,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }
    
    if (nbr_back == MPI_PROC_NULL) {
        int gk = 0;
        for (int i = 1; i <= local_nx; ++i)
            for (int j = 1; j <= local_ny; ++j) {
                int glob_k_use = std::max(0, start_z - 1);
                double x = (start_x + (i-1)) * dx;
                double y = (start_y + (j-1)) * dy;
                double z = glob_k_use * dz;
                u0[idx(i,j,gk,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }
    if (nbr_front == MPI_PROC_NULL) {
        int gk = gz-1;
        for (int i = 1; i <= local_nx; ++i)
            for (int j = 1; j <= local_ny; ++j) {
                int glob_k_use = std::min(Nz-1, start_z + (local_nz));
                double x = (start_x + (i-1)) * dx;
                double y = (start_y + (j-1)) * dy;
                double z = glob_k_use * dz;
                u0[idx(i,j,gk,gx,gy,gz)] = an_sol(x,y,z,0.0);
            }
    }

    double global_max_err0 = 0.0;
    MPI_Reduce(&local_max_err0, &global_max_err0, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (world_rank == 0) std::cout << "Max error at timestep 0 = " << global_max_err0 << std::endl;
    cudaMalloc(&dev_u1, total*sizeof(double));
    double global_max_err1 = 0.0;
    double local_max_err1 = 0.0;

    it1_kernel_launcher(
        dev_u0, dev_u1,
        local_nx, local_ny, local_nz,
        start_x, start_y, start_z,
        dx, dy, dz, a, tau,
        local_max_err1
    );
    
    MPI_Reduce(&local_max_err1, &global_max_err1, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (world_rank == 0) std::cout << "Max error at timestep 1 = " << global_max_err1 << std::endl;
    cudaMemcpy(u1.data(), dev_u1, total*sizeof(double), cudaMemcpyDeviceToHost);
  
    int size_x_face = local_ny * local_nz;
    int size_y_face = local_nx * local_nz;
    int size_z_face = local_nx * local_ny;

    std::vector<double> sbuf_x_left(size_x_face), rbuf_x_left(size_x_face);
    std::vector<double> sbuf_x_right(size_x_face), rbuf_x_right(size_x_face);

    std::vector<double> sbuf_y_down(size_y_face), rbuf_y_down(size_y_face);
    std::vector<double> sbuf_y_up(size_y_face),   rbuf_y_up(size_y_face);

    std::vector<double> sbuf_z_back(size_z_face), rbuf_z_back(size_z_face);
    std::vector<double> sbuf_z_front(size_z_face),rbuf_z_front(size_z_face);

    cudaMalloc(&dev_u, total*sizeof(double));    
    for (int n = 2; n < Nt; ++n) {
        double t = n * tau;
        if (n > 2){
            cudaMemcpy(u0.data(), dev_u0, total*sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(u1.data(), dev_u1, total*sizeof(double), cudaMemcpyDeviceToHost);
        }
        
        MPI_Request reqs[12];
        int reqcnt = 0;

       
        if (nbr_left != MPI_PROC_NULL) {
            int pos = 0;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k)
                    sbuf_x_left[pos++] = u1[idx(1,j,k,gx,gy,gz)];
            MPI_Irecv(rbuf_x_left.data(), size_x_face, MPI_DOUBLE, nbr_left, 101, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_x_left.data(), size_x_face, MPI_DOUBLE, nbr_left, 102, cart_comm, &reqs[reqcnt++]);
        }
        if (nbr_right != MPI_PROC_NULL) {
            int pos = 0;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k)
                    sbuf_x_right[pos++] = u1[idx(local_nx,j,k,gx,gy,gz)];
            MPI_Irecv(rbuf_x_right.data(), size_x_face, MPI_DOUBLE, nbr_right, 102, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_x_right.data(), size_x_face, MPI_DOUBLE, nbr_right, 101, cart_comm, &reqs[reqcnt++]);
        }

       
        if (nbr_down != MPI_PROC_NULL) {
            int pos = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k)
                    sbuf_y_down[pos++] = u1[idx(i,1,k,gx,gy,gz)];
            MPI_Irecv(rbuf_y_down.data(), size_y_face, MPI_DOUBLE, nbr_down, 201, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_y_down.data(), size_y_face, MPI_DOUBLE, nbr_down, 202, cart_comm, &reqs[reqcnt++]);
        }
        if (nbr_up != MPI_PROC_NULL) {
            int pos = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k)
                    sbuf_y_up[pos++] = u1[idx(i,local_ny,k,gx,gy,gz)];
            MPI_Irecv(rbuf_y_up.data(), size_y_face, MPI_DOUBLE, nbr_up, 202, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_y_up.data(), size_y_face, MPI_DOUBLE, nbr_up, 201, cart_comm, &reqs[reqcnt++]);
        }

        
        if (nbr_back != MPI_PROC_NULL) {
            int pos = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j)
                    sbuf_z_back[pos++] = u1[idx(i,j,1,gx,gy,gz)];
            MPI_Irecv(rbuf_z_back.data(), size_z_face, MPI_DOUBLE, nbr_back, 301, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_z_back.data(), size_z_face, MPI_DOUBLE, nbr_back, 302, cart_comm, &reqs[reqcnt++]);
        }
        if (nbr_front != MPI_PROC_NULL) {
            int pos = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j)
                    sbuf_z_front[pos++] = u1[idx(i,j,local_nz,gx,gy,gz)];
            MPI_Irecv(rbuf_z_front.data(), size_z_face, MPI_DOUBLE, nbr_front, 302, cart_comm, &reqs[reqcnt++]);
            MPI_Isend(sbuf_z_front.data(), size_z_face, MPI_DOUBLE, nbr_front, 301, cart_comm, &reqs[reqcnt++]);
        }

        
        if (reqcnt) MPI_Waitall(reqcnt, reqs, MPI_STATUSES_IGNORE);

        
        if (nbr_left != MPI_PROC_NULL) {
            int pos = 0;
            int gi = 0;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k)
                    u1[idx(gi,j,k,gx,gy,gz)] = rbuf_x_left[pos++];
        } else {
           
            int gi = 0;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k) {
                    int glob_i_use = std::max(0, start_x - 1);
                    double x = glob_i_use * dx;
                    double y = (start_y + (j-1)) * dy;
                    double z = (start_z + (k-1)) * dz;
                    u1[idx(gi,j,k,gx,gy,gz)] = an_sol(x,y,z,t - tau); 
                }
        }
        if (nbr_right != MPI_PROC_NULL) {
            int pos = 0;
            int gi = gx-1;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k)
                    u1[idx(gi,j,k,gx,gy,gz)] = rbuf_x_right[pos++];
        } else {
            int gi = gx-1;
            for (int j = 1; j <= local_ny; ++j)
                for (int k = 1; k <= local_nz; ++k) {
                    int glob_i_use = std::min(Nx-1, start_x + local_nx);
                    double x = glob_i_use * dx;
                    double y = (start_y + (j-1)) * dy;
                    double z = (start_z + (k-1)) * dz;
                    u1[idx(gi,j,k,gx,gy,gz)] = an_sol(x,y,z,t - tau);
                }
        }

        
        if (nbr_down != MPI_PROC_NULL) {
            int pos = 0;
            int gj = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k)
                    u1[idx(i,gj,k,gx,gy,gz)] = rbuf_y_down[pos++];
        } else {
            int gj = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k) {
                    int glob_j_use = std::max(0, start_y - 1);
                    double x = (start_x + (i-1)) * dx;
                    double y = glob_j_use * dy;
                    double z = (start_z + (k-1)) * dz;
                    u1[idx(i,gj,k,gx,gy,gz)] = an_sol(x,y,z,t - tau);
                }
        }
        if (nbr_up != MPI_PROC_NULL) {
            int pos = 0;
            int gj = gy-1;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k)
                    u1[idx(i,gj,k,gx,gy,gz)] = rbuf_y_up[pos++];
        } else {
            int gj = gy-1;
            for (int i = 1; i <= local_nx; ++i)
                for (int k = 1; k <= local_nz; ++k) {
                    int glob_j_use = std::min(Ny-1, start_y + local_ny);
                    double x = (start_x + (i-1)) * dx;
                    double y = glob_j_use * dy;
                    double z = (start_z + (k-1)) * dz;
                    u1[idx(i,gj,k,gx,gy,gz)] = an_sol(x,y,z,t - tau);
                }
        }

       
        if (nbr_back != MPI_PROC_NULL) {
            int pos = 0;
            int gk = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j)
                    u1[idx(i,j,gk,gx,gy,gz)] = rbuf_z_back[pos++];
        } else {
            int gk = 0;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j) {
                    int glob_k_use = std::max(0, start_z - 1);
                    double x = (start_x + (i-1)) * dx;
                    double y = (start_y + (j-1)) * dy;
                    double z = glob_k_use * dz;
                    u1[idx(i,j,gk,gx,gy,gz)] = an_sol(x,y,z,t - tau);
                }
        }
        if (nbr_front != MPI_PROC_NULL) {
            int pos = 0;
            int gk = gz-1;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j)
                    u1[idx(i,j,gk,gx,gy,gz)] = rbuf_z_front[pos++];
        } else {
            int gk = gz-1;
            for (int i = 1; i <= local_nx; ++i)
                for (int j = 1; j <= local_ny; ++j) {
                    int glob_k_use = std::min(Nz-1, start_z + local_nz);
                    double x = (start_x + (i-1)) * dx;
                    double y = (start_y + (j-1)) * dy;
                    double z = glob_k_use * dz;
                    u1[idx(i,j,gk,gx,gy,gz)] = an_sol(x,y,z,t - tau);
                }
        }

        cudaMemcpy(dev_u0, u0.data(), total*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_u1, u1.data(), total*sizeof(double), cudaMemcpyHostToDevice);
        double local_max_err = 0.0;
        
        main_kernel_launcher(
            dev_u0, dev_u1, dev_u,
            local_nx, local_ny, local_nz,
            Nx, Ny, Nz,
            start_x, start_y, start_z,
            dx, dy, dz,
            a, tau, t,
            local_max_err
        );

        
        double global_max_err = 0.0;
        MPI_Reduce(&local_max_err, &global_max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (world_rank == 0) std::cout << "Max error at timestep " << n << " = " << global_max_err << std::endl;
  
        
        std::swap(dev_u0, dev_u1);
        std::swap(dev_u1, dev_u);
    }

    double t_end = MPI_Wtime();
    double elapsed = t_end - t_beg;
    double max_elapsed;
    MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (world_rank == 0) std::cout << "Execution time (max over ranks): " << max_elapsed << " s" << std::endl;
    
    cudaFree(dev_u0);
    cudaFree(dev_u1);
    cudaFree(dev_u);
    MPI_Finalize();
    return 0;
}
