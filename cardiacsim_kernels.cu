#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 16

double *flatten_matrix(double **mat, int width, int height);
double **unflatten_matrix(double *flat, int width, int height);
int mat_equal(double **m1, double **m2, int width, int height);

__global__ void v1_PDE(double **E, double **E_prev, double **R,
					   const double alpha, const int n, const int m, const double kk,
					   const double dt, const double a, const double epsilon,
					   const double M1, const double M2, const double b)
{

	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	//Load ghost cells (halos)
	if (row < 1)
	{
		E_prev[row][0] = E_prev[row][2];
		E_prev[row][n + 1] = E_prev[row][n - 1];
	}

	if (col < 1)
	{
		E_prev[0][col] = E_prev[2][col];
		E_prev[m + 1][col] = E_prev[m - 1][col];
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

	double **d_E, **d_R, **d_E_prev;
	int nx = n + 2, ny = m + 2;
	int matSize = sizeof(double *) * ny + sizeof(double) * nx * ny;

	cudaMalloc(&d_E, matSize);
	cudaMalloc(&d_R, matSize);
	cudaMalloc(&d_E_prev, matSize);

	cudaMemcpy(d_E, E, matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_prev, E_prev, matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, R, matSize, cudaMemcpyHostToDevice);

	v1_PDE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();
	v1_ODE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();

	cudaMemcpy(E, d_E, matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(R, d_R, matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(E_prev, d_E_prev, matSize, cudaMemcpyDeviceToHost);

	cudaFree(d_E);
	cudaFree(d_R);
	cudaFree(d_E_prev);
}

double *flatten_matrix(double **mat, int width, int height)
{
	double *flattened = (double *)malloc(width * height * sizeof(double));
	assert(flattened);

	for (int i = 0; i < width * height; i++)
	{
		int row = i / width;
		int col = i % width;
		flattened[i] = mat[row][col];
	}

	return flattened;
}

double **unflatten_matrix(double *flat, int width, int height)
{
	double **mat = (double **)malloc(height * sizeof(double *));
	assert(mat);

	for (int i = 0; i < height; i++)
	{
		mat[i] = (double *)malloc(width * sizeof(double));
		memcpy(mat[i], flat + i * width, width * sizeof(double));
	}

	return mat;
}

int mat_equal(double **m1, double **m2, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (m2[i][j] != m1[i][j])
			{
				printf("i: %d, j: %d m1=%f, m2=%f\n", i, j, m1[i][j], m2[i][j]);
				return 0;
			}
		}
	}
	return 1;
}