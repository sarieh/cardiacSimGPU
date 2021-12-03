#include <stdio.h>
#include <assert.h>

#define BLOCK_SIZE 16

double *d_E, *d_R, *d_E_prev;

int mat_equal(double **m1, double **m2, int width, int height);

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

void kernel1(double *E, double *E_prev, double *R, const double alpha, const int n, const int m, const double kk,
			 const double dt, const double a, const double epsilon, const double M1, const double M2, const double b, int shouldMalloc, int shouldFree)
{

	/// TODO: put ghost stuff in method
	int width = m + 2;
	for (int j = 1; j <= m; j++)
	{
		E_prev[j * width] = E_prev[j * width + 2];
		E_prev[j * width + n + 1] = E_prev[j * width + n - 1];
	}

	for (int i = 1; i <= n; i++)
	{
		E_prev[i] = E_prev[2 * width + i];
		E_prev[(m + 1) * width + i] = E_prev[(m - 1) * width + i];
	}

	int nx = n + 2, ny = m + 2;

	int matSize = sizeof(double) * nx * ny;

	if (shouldMalloc)
	{
		cudaMalloc(&d_E, matSize);
		cudaMalloc(&d_R, matSize);
		cudaMalloc(&d_E_prev, matSize);
	}

	cudaMemcpy(d_E, &E[0], matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_E_prev, &E_prev[0], matSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_R, &R[0], matSize, cudaMemcpyHostToDevice);

	const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
	int dimension = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	const dim3 grid(dimension, dimension);

	v1_PDE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();
	v1_ODE<<<grid, block>>>(d_E, d_E_prev, d_R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
	cudaDeviceSynchronize();

	cudaMemcpy(E, d_E, matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(R, d_R, matSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(E_prev, d_E_prev, matSize, cudaMemcpyDeviceToHost);
	if (shouldFree)
	{
		cudaFree(d_E);
		cudaFree(d_R);
		cudaFree(d_E_prev);
	}
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

void unflatten_matrix(double **dest, double *flat, int width, int height)
{

	for (int i = 0; i < height; i++)
	{
		dest[i] = (double *)malloc(width * sizeof(double));
		memcpy(dest[i], flat + i * width, width * sizeof(double));
	}
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