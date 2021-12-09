#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 16

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
						  const double M1, const double M2, const double b, int bx, int by)
{
	extern __shared__ double shared_E_prev[];

	const int block_width = blockDim.x + 2;

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int l_col = threadIdx.x + 1;
	int l_row = threadIdx.y + 1;

	int gwidth = m + 2;
	int gindex = row * gwidth + col;

	int lindex = l_row * block_width + l_col;

	// Read input elements into shared memory
	shared_E_prev[lindex] = E_prev[gindex];
	//Load ghost cells
	if (threadIdx.x < 1)
	{
		shared_E_prev[lindex - 1] = E_prev[gindex - 1];
		shared_E_prev[lindex + bx] = E_prev[gindex + bx];
	}
	if (threadIdx.y < 1)
	{
		shared_E_prev[lindex - block_width] = E_prev[gindex - gwidth];
		shared_E_prev[lindex + block_width * by] = E_prev[gindex + gwidth * by];
	}

	__syncthreads(); // Make sure all threads loaded into the shared memory

	if (row <= m && col <= n)
	{
		//PDE
		double e_current = shared_E_prev[lindex] + alpha * (shared_E_prev[lindex + 1] + shared_E_prev[lindex - 1] - 4 * shared_E_prev[lindex] + shared_E_prev[lindex + block_width] + shared_E_prev[lindex - block_width]);
		double r_current = R[gindex];
		__syncthreads();

		//ODE
		e_current = e_current - dt * (kk * e_current * (e_current - a) * (e_current - 1) + e_current * r_current);
		r_current = r_current + dt * (epsilon + M1 * r_current / (e_current + M2)) * (-r_current - kk * e_current * (e_current - b - 1));

		E[gindex] = e_current;
		R[gindex] = r_current;
	}
}


__global__ void halos_kernel(double *E_prev, const int m){

	int x = threadIdx.x + 1;
	int width = m + 2;
	
	E_prev[x] = E_prev[x + 2*width];
	E_prev[x + (m+1)*width] = E_prev[x + (m-1)*width];
	E_prev[x*width] = E_prev[x*width + 2];
	E_prev[x*width + (m + 1)] = E_prev[x*width + (m - 1)];
}

void deviceKernel(double **E, double **E_prev, double **R, double **d_E, double **d_E_prev, double **d_R, const double alpha, const int n, const int m, const double kk,
	const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, int shouldMalloc, int shouldFree, int v, int swap, int bx, int by)
{

	int nx = n + 2, ny = m + 2;
	int matSize = sizeof(double) * nx * ny;
	int copyOffset = ny;

	if (shouldMalloc)
	{
		cudaMalloc(&(*d_E), matSize);
		cudaMalloc(&(*d_R), matSize);
		cudaMalloc(&(*d_E_prev), matSize);
		cudaMemcpy(*d_R, &R[copyOffset], matSize, cudaMemcpyHostToDevice);
		cudaMemcpy(*d_E, &E[copyOffset], matSize, cudaMemcpyHostToDevice);
		cudaMemcpy(*d_E_prev, &E_prev[copyOffset], matSize, cudaMemcpyHostToDevice);
	}

	const dim3 block(bx, by);
	int dimension_x = (n + bx - 1) / bx;
	int dimension_y = (n + by - 1) / by;
	const dim3 grid(dimension_x, dimension_y);

	const int block_width = bx + 2;
	const int block_height = by + 2;
	const int sharedBlockSize = block_width * block_height * sizeof(double);
    
	if(swap % 2)
		halos_kernel<<<1, m>>>(*d_E, m);
	else
		halos_kernel<<<1, m>>>(*d_E_prev, m);
		
	cudaDeviceSynchronize();

	if (v == 1)
	{
		if(swap % 2)
			v1_PDE<<<grid, block>>>(*d_E_prev, *d_E, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		else
			v1_PDE<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);

		cudaDeviceSynchronize();
		if(swap % 2)
			v1_ODE<<<grid, block>>>(*d_E_prev, *d_E, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		else
			v1_ODE<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else if (v == 2)
	{
		if(swap % 2)
			v2_kernel<<<grid, block>>>(*d_E_prev, *d_E, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		else
			v2_kernel<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else if (v == 3)
	{
		if(swap % 2)
			v3_kernel<<<grid, block>>>(*d_E_prev, *d_E, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		else
			v3_kernel<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else
	{
		if(swap % 2)
			v4_kernel<<<grid, block, sharedBlockSize>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, bx, by);
		else
			v4_kernel<<<grid, block, sharedBlockSize>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, bx, by);
		
	}

	cudaDeviceSynchronize();
	if (shouldFree)
	{
		cudaMemcpy(&E[copyOffset] , *d_E, matSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(&E_prev[copyOffset] , *d_E_prev, matSize, cudaMemcpyDeviceToHost);	
		cudaMemcpy(&R[copyOffset] , *d_R, matSize, cudaMemcpyDeviceToHost);
		cudaFree(*d_E);
		cudaFree(*d_R);
		cudaFree(*d_E_prev);
	}
}
