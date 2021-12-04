#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 16
#define TWO 2

double *d_E, *d_R, *d_E_prev;

void mirror_halos(double **mat, int m, int n);

__global__ void v1_PDE(double *E, double *E_prev, double *R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int width = m + 2;

	int index = row * width + col;

	if (row <= m && col <= n)
		E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + width] + E_prev[index - width]);
}

__global__ void v1_ODE(double *E, double *E_prev, double *R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int index = row * (m + 2) + col;

	if (row <= m && col <= n)
	{
		E[index] = E[index] - dt * (kk * E[index] * (E[index] - a) * (E[index] - 1) + E[index] * R[index]);
		R[index] = R[index] + dt * (epsilon + M1 * R[index] / (E[index] + M2)) * (-R[index] - kk * E[index] * (E[index] - b - 1));
	}
}

__global__ void v2_kernel(double *E, double *E_prev, double *R,
						  const double alpha, const int n, const int m, const double kk,
						  const double dt, const double a, const double epsilon,
						  const double M1, const double M2, const double b)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int width = m + 2;

	int index = row * width + col;

	if (row <= m && col <= n)
	{
		E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + width] + E_prev[index - width]);

		__syncthreads();

		E[index] = E[index] - dt * (kk * E[index] * (E[index] - a) * (E[index] - 1) + E[index] * R[index]);
		R[index] = R[index] + dt * (epsilon + M1 * R[index] / (E[index] + M2)) * (-R[index] - kk * E[index] * (E[index] - b - 1));
	}
}

__global__ void v3_kernel(double *E, double *E_prev, double *R,
						  const double alpha, const int n, const int m, const double kk,
						  const double dt, const double a, const double epsilon,
						  const double M1, const double M2, const double b)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int width = m + 2;

	int index = row * width + col;

	if (row <= m && col <= n)
	{
		E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + width] + E_prev[index - width]);

		__syncthreads();

		double e_current = E[index];
		double r_current = R[index];

		E[index] = e_current - dt * (kk * e_current * (e_current - a) * (e_current - 1) + e_current * r_current);
		e_current = E[index];

		R[index] = r_current + dt * (epsilon + M1 * r_current / (e_current + M2)) * (-r_current - kk * e_current * (e_current - b - 1));
	}
}

__global__ void v4_kernel(double *E, double *E_prev, double *R,
	const double alpha, const int n, const int m, const double kk,
	const double dt, const double a, const double epsilon,
	const double M1, const double M2, const double b)
{
	// const int tmpSize = BLOCK_SIZE*BLOCK_SIZE + (TWO * m) + (n * TWO);
	__shared__ int shared_E[1056];
	__shared__ int shared_E_prev[1056];
	__shared__ int shared_R[1056];

	int l_col = threadIdx.x + 1;
	int l_row = threadIdx.y + 1;
	int lwidth = BLOCK_SIZE + 2;
	int lindex = l_row * lwidth + l_col;

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int gwidth = m + 2;
	int gindex = row * gwidth + col;
	
	shared_E[lindex] = E[gindex];
	shared_E_prev[lindex] = E_prev[gindex];
	shared_R[lindex] = R[gindex];

	if (threadIdx.x < 1) {
		shared_E_prev[lindex - 1] = E_prev[gindex - 1];
		shared_E_prev[lindex + BLOCK_SIZE] = E_prev[gindex + BLOCK_SIZE];
	}

	if (threadIdx.y < 1) {
		shared_E_prev[lindex - lwidth] = E_prev[gindex - gwidth];
		shared_E_prev[lindex + lwidth*m] = E_prev[gindex + gwidth*m];
	}
		
	__syncthreads();

	if (row <= m && col <= n)
	{
		shared_E[lindex] = shared_E_prev[lindex] + alpha * (shared_E_prev[lindex + 1] + shared_E_prev[lindex - 1] - 4 * shared_E_prev[lindex] + shared_E_prev[lindex + lwidth] + shared_E_prev[lindex - lwidth]);

		__syncthreads();

		double e_current = shared_E[lindex];
		double r_current = shared_R[lindex];

		shared_E[lindex] = e_current - dt * (kk * e_current * (e_current - a) * (e_current - 1) + e_current * r_current);
		e_current = shared_E[lindex];

		shared_R[lindex] = r_current + dt * (epsilon + M1 * r_current / (e_current + M2)) * (-r_current - kk * e_current * (e_current - b - 1));
	}
}

void deviceKernel(double **E, double **E_prev, double **R, const double alpha, const int n, const int m, const double kk,
	const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, int shouldMalloc, int shouldFree, int v){
	
	mirror_halos(E_prev, m, n);

	int nx = n + 2, ny = m + 2;
	int matSize = sizeof(double) * nx * ny;
	int copyOffset = ny;

	if (shouldMalloc)
	{
		cudaMalloc(&d_E, matSize);
		cudaMalloc(&d_R, matSize);
		cudaMalloc(&d_E_prev, matSize);
		cudaMemcpy(d_R, &R[0] + copyOffset, matSize, cudaMemcpyHostToDevice);
	}

	cudaMemcpy(d_E, &E[0] + copyOffset, matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_prev, &E_prev[0] + copyOffset, matSize, cudaMemcpyHostToDevice);

	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	int dimension = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const dim3 grid(dimension, dimension);

	if(v == 1){
		v1_PDE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();
		v1_ODE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();		
	}else if(v == 2){
		v2_kernel<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();			
	}else if(v == 3){
		v3_kernel<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();
	}else{
		v4_kernel<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();
	}
	
	// printf("%d \n", BLOCK_SIZE*BLOCK_SIZE + (2 * m) + (n * 2));
	cudaMemcpy(E + copyOffset, d_E, matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(E_prev + copyOffset, d_E_prev, matSize, cudaMemcpyDeviceToHost);
	if (shouldFree)
	{
		cudaMemcpy(R + copyOffset, d_R, matSize, cudaMemcpyDeviceToHost);
		cudaFree(d_E);
		cudaFree(d_R);
		cudaFree(d_E_prev);
	}
}

void mirror_halos(double **mat, int m, int n)
{
	int i, j;

	for (j = 1; j <= m; j++)
	{
		mat[j][0] = mat[j][2];
		mat[j][n + 1] = mat[j][n - 1];
	}

	for (i = 1; i <= n; i++)
	{
		mat[0][i] = mat[2][i];
		mat[m + 1][i] = mat[m - 1][i];
	}
}