#include <stdio.h>

#define BLOCK_SIZE 16

__global__ void v1_PDE(double **E, double **E_prev, double **R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{
		

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//Load ghost cells (halos)
	if (row < 1) {
		E_prev[row][0] = E_prev[row][2];
		E_prev[row][n+1] = E_prev[row][n-1];
	}

	if (col < 1) {
		E_prev[0][col] = E_prev[2][col];
		E_prev[m+1][col] = E_prev[m-1][col];
	}

	__syncthreads();
	E[row][col] = E_prev[row][col] + alpha * (E_prev[row][col + 1] + E_prev[row][col - 1] - 4 * E_prev[row][col] + E_prev[row + 1][col] + E_prev[row - 1][col]);
}

__global__ void v1_ODE(double **E, double **E_prev, double **R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	E[row][col] = E[row][col] - dt * (kk * E[row][col] * (E[row][col] - a) * (E[row][col] - 1) + E[row][col] * R[row][col]);
	R[row][col] = R[row][col] + dt * (epsilon + M1 * R[row][col] / (E[row][col] + M2)) * (-R[row][col] - kk * E[row][col] * (E[row][col] - b - 1));
}

void kernel1(double **E, double **E_prev, double **R, const double alpha, const int n, const int m, const double kk,
					const double dt, const double a, const double epsilon, const double M1, const double M2, const double b)
{
	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	int dimension = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const dim3 grid(dimension, dimension);

	double **E_d, **R_d, **E_prev_d;
	int nx = n + 2, ny = m + 2;
	int matSize = sizeof(double *) * ny + sizeof(double) * nx * ny;
	
	cudaMalloc( &E_d,  matSize ) ;
	cudaMalloc( &R_d,  matSize ) ;
	cudaMalloc( &E_prev_d,  matSize ) ;
	
	cudaMemcpy( E_d, E, matSize, cudaMemcpyHostToDevice) ;
	cudaMemcpy( E_prev_d, E_prev, matSize, cudaMemcpyHostToDevice) ;
	cudaMemcpy( R_d, R, matSize, cudaMemcpyHostToDevice) ;

	v1_PDE<<<grid, block>>>(E_d, E_prev_d, R_d, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();
	v1_ODE<<<grid, block>>>(E_d, E_prev_d, R_d, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();

	cudaMemcpy(E , E_d, matSize, cudaMemcpyDeviceToHost) ;
	cudaMemcpy(R , R_d, matSize, cudaMemcpyDeviceToHost) ;
	cudaMemcpy(E_prev , E_prev_d, matSize, cudaMemcpyDeviceToHost) ;
	
	cudaFree(E_d);
	cudaFree(R_d);
	cudaFree(E_prev_d);
}