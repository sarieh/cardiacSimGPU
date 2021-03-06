
#ifndef __CARDIACSIM_H__
#define __CARDIACSIM_H__

void deviceKernel(double **E, double **E_prev, double **R, double **d_E, double **d_E_prev, double **d_R, const double alpha,
				  const int n, const int m, const double kk, const double dt, const double a, const double epsilon, const double M1, const double M2,
				  const double b, int shouldFree, int v, int swap, int bx, int by, int plot);

void copyDataHostToDevice(double **E, double **E_prev, double **R, double **d_E, double **d_E_prev, double **d_R, const int n, const int m);

#endif