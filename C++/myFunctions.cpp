#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void CPUBenchmarkSC(float** A, float* x, float* b, int k, int N) 
{
    for (int i = 0; i < N; i++) {
        b[i] = 0.0;
        for (int j = 0; j < N; j++) {
            b[i] += k * A[i][j] * x[j];
        }
    }
    b[0] = 1 * b[0];
}


void CPUBenchmarkMC(float** A, float* x, float* b, int k, int N)
{
#pragma omp parallel
    {
        int i, j;
#pragma omp for
        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                b[i] += k * A[i][j] * x[j];
            }
        }
    }
    b[0] = 1*b[0];
}