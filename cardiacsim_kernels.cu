#include <stdio.h>
#include <assert.h>

void deviceKernelWithSwappedData(double **E, double **E_prev, double **d_E, double **d_E_prev, double **d_R, const double alpha, const int n, const int m, const double kk,
								 const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, int v, int bx, int by, int plot);

void copyBack(double **E, double **E_prev, double **d_E, double **d_E_prev, int swap, int matSize, int copyOffset, int copyEprev);

__global__ void v1_PDE(double *E, double *E_prev, double *R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int width = m + 2;

	int index = row * width + col;

	if (row <= m && col <= n) // thread inside the matrix
		E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + width] + E_prev[index - width]);
}

__global__ void v1_ODE(double *E, double *E_prev, double *R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	int index = row * (m + 2) + col; // m + 2 to account for halos

	if (row <= m && col <= n)
	{ // thread inside the matrix
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

	int width = m + 2; // +2 to account for halos

	int index = row * width + col;

	if (row <= m && col <= n) // thread inside the matrix
	{
		E[index] = E_prev[index] + alpha * (E_prev[index + 1] + E_prev[index - 1] - 4 * E_prev[index] + E_prev[index + width] + E_prev[index - width]);

		__syncthreads(); // barrier before the execution of the ODE

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

		double e_current = E[index]; // store the value of E in the register to avoid global memory references
		double r_current = R[index]; // store the value of R in the register to avoid global memory references

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
	// shared memory tile with dynamic size
	extern __shared__ double shared_E_prev[];

	const int block_width = blockDim.x + 2;

	// global row and column
	int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int row = blockIdx.y * blockDim.y + threadIdx.y + 1;

	// local (tile) row and column
	int l_col = threadIdx.x + 1;
	int l_row = threadIdx.y + 1;

	// global width and index
	int gwidth = m + 2;
	int gindex = row * gwidth + col;

	// local (tile) index
	int lindex = l_row * block_width + l_col;

	// Read input elements into shared memory
	shared_E_prev[lindex] = E_prev[gindex];
	//Load ghost cells into tile
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

	__syncthreads(); // Make sure all threads are done loading into the shared memory

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

__global__ void halos_kernel(double *E_prev, const int m)
{

	int x = threadIdx.x + 1;
	int width = m + 2;
	// fill halos at the top, bottom, rightmost, and leftmost cells
	E_prev[x] = E_prev[x + 2 * width];
	E_prev[x + (m + 1) * width] = E_prev[x + (m - 1) * width];
	E_prev[x * width] = E_prev[x * width + 2];
	E_prev[x * width + (m + 1)] = E_prev[x * width + (m - 1)];
}

void deviceKernel(double **E, double **E_prev, double **R, double **d_E, double **d_E_prev, double **d_R, const double alpha,
				  const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2,
				  const double b, int shouldFree, int v, int swap, int bx, int by, int plot)
{
	int nx = n + 2, ny = m + 2; // dimensions taking into account halo cells
	int matSize = sizeof(double) * nx * ny; // size of the matrix (for memory management)
	int copyOffset = ny; // offset to start copying after the pointers, starting at the actual matrix items

	if (swap % 2) // pass the matrices swapped
		deviceKernelWithSwappedData(E, E_prev, d_E_prev, d_E, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, v, bx, by, plot);
	else // pass the matrices without swapping
		deviceKernelWithSwappedData(E, E_prev, d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, v, bx, by, plot);

	if (plot) // copy back for plotting
		copyBack(E, E_prev, d_E, d_E_prev, swap, matSize, copyOffset, 0);

	if (shouldFree)
	{ // should free - final iteration of the simulation
		// copy back R, E, and E_prev
		copyBack(E, E_prev, d_E, d_E_prev, swap, matSize, copyOffset, 1);
		cudaMemcpy(&R[copyOffset], *d_R, matSize, cudaMemcpyDeviceToHost);
		// free device allocated memory
		cudaFree(*d_E);
		cudaFree(*d_R);
		cudaFree(*d_E_prev);
	}
}

void deviceKernelWithSwappedData(double **E, double **E_prev, double **d_E, double **d_E_prev, double **d_R, const double alpha, const int n, const int m, const double kk,
								 const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, int v, int bx, int by, int plot)
{

	// block and grid sizes for the simulation kernels
	const dim3 block(bx, by);
	int dimension_x = (n + bx - 1) / bx;
	int dimension_y = (n + by - 1) / by;
	const dim3 grid(dimension_x, dimension_y);
	// block sizes accounting for halos
	const int block_width = bx + 2;
	const int block_height = by + 2;
	// define the size of the tile to match the block size
	const int sharedBlockSize = block_width * block_height * sizeof(double);
	// halo mirroring kernel
	halos_kernel<<<1, m>>>(*d_E_prev, m);
	cudaDeviceSynchronize(); // make sure mirroring is done

	// run the appropriate kernel version
	if (v == 1)
	{
		v1_PDE<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
		cudaDeviceSynchronize();
		v1_ODE<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else if (v == 2)
	{
		v2_kernel<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else if (v == 3)
	{
		v3_kernel<<<grid, block>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	}
	else
	{
		v4_kernel<<<grid, block, sharedBlockSize>>>(*d_E, *d_E_prev, *d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b, bx, by);
	}
	cudaDeviceSynchronize(); // make sure the kernel is done executing
}

void copyDataHostToDevice(double **E, double **E_prev, double **R, double **d_E, double **d_E_prev, double **d_R, const int n, const int m)
{ // allocate device memory and copy host data into it
	int nx = n + 2, ny = m + 2;
	int matSize = sizeof(double) * nx * ny;
	int copyOffset = ny;

	cudaMalloc(&(*d_E), matSize);
	cudaMalloc(&(*d_R), matSize);
	cudaMalloc(&(*d_E_prev), matSize);
	cudaMemcpy(*d_R, &R[copyOffset], matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(*d_E, &E[copyOffset], matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(*d_E_prev, &E_prev[copyOffset], matSize, cudaMemcpyHostToDevice);
}

void copyBack(double **E, double **E_prev, double **d_E, double **d_E_prev, int swap, int matSize, int copyOffset, int copyEprev)
{
	if (swap % 2)
	{ // copy the matrices without swapping
		cudaMemcpy(&E[copyOffset], *d_E, matSize, cudaMemcpyDeviceToHost);
		if (copyEprev) // copy back E_prev if needed
			cudaMemcpy(&E_prev[copyOffset], *d_E_prev, matSize, cudaMemcpyDeviceToHost);
	}
	else
	{ // copy the matrices with swapping
		cudaMemcpy(&E[copyOffset], *d_E_prev, matSize, cudaMemcpyDeviceToHost);
		if (copyEprev)
			cudaMemcpy(&E_prev[copyOffset], *d_E, matSize, cudaMemcpyDeviceToHost);
	}
}