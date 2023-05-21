// C++.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "myFunctions.h"
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main()
{
    int i, N, times;
    float** A, * x, * b;
    clock_t start, end;
    double cpu_time_used;

    N = 5000;
    times = 10000;

    A = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        A[i] = (float*)calloc(N, sizeof(float));
    }
    x = (float*)calloc(N, sizeof(float));
    b = (float*)calloc(N, sizeof(float));

    // Initialize A and x with random values
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / 10.0);
        }
        x[i] = (float)rand() / (float)(RAND_MAX / 10.0);
    }

    start = clock();
    for (i = 0; i < times; i++) {
        CPUBenchmarkSC(A, x, b, i, N);
    }
    end = clock();
    cpu_time_used = difftime(end, start) / CLOCKS_PER_SEC;
    printf("single core cpu time used: %f seconds\n", cpu_time_used);

    start = clock();
    for (i = 0; i < times; i++) {
        CPUBenchmarkMC(A, x, b, i, N);
    }
    end = clock();
    cpu_time_used = difftime(end, start) / CLOCKS_PER_SEC;
    printf("Multiple core CPU time used: %f seconds\n", cpu_time_used);

    return 0;
}