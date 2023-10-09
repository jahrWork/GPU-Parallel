#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define ROWS 5000
#define COLS 5000
#define TIMES 10000

void CPUBenchmark(float** A, float* x, float* b, int k)
{
#pragma omp parallel
    {
        int i, j;
#pragma omp for
        for (i = 0; i < COLS; i++) {
            for (j = 0; j < COLS; j++) {
                b[i] += k * A[i][j] * x[j];
            }
        }
    }
    b[0] = 1 * b[0];
}




int main() {

   
    int i;
    float** A, * x, * b;
    clock_t start, end;
    double cpu_time_used;

    A = (float**)malloc(ROWS * sizeof(float*));
    for (int i = 0; i < ROWS; i++) {
        A[i] = (float*)calloc(COLS, sizeof(float));
    }
    x = (float*)calloc(COLS, sizeof(float));
    b = (float*)calloc(ROWS, sizeof(float));

    // Initialize A and x with random values
    srand(time(NULL));
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            A[i][j] = (float)rand() / (float)(RAND_MAX / 10.0);
        }
        x[i] = (float)rand() / (float)(RAND_MAX / 10.0);
    }

   
    // Add the following lines before the start of the benchmark:
    printf("Press Enter to start the first benchmark...\n");
    getchar();

    start = clock();
    for (i = 0; i < TIMES; i++) {
        CPUBenchmark(A, x, b, i);
    }
    end = clock();
    cpu_time_used = difftime(end, start) / CLOCKS_PER_SEC;
    printf("Single core CPU time used: %f seconds\n", cpu_time_used);

    // Add the following lines before the start of the benchmark:
    printf("Press Enter to start the second core benchmark...\n");
    getchar();

    start = clock();
    for (i = 0; i < TIMES; i++) {
        CPUBenchmark(A, x, b, i);
    }
    end = clock();
    cpu_time_used = difftime(end, start) / CLOCKS_PER_SEC;
    printf("Multiple core CPU time used: %f seconds\n", cpu_time_used);


    return 0;

}




//#include <stdio.h>
//#include <stdlib.h>
//#include <time.h>
//#include <iostream>
//
//#define ROWS 5000
//#define COLS 5000
//#define TIMES 10000
//
//// Para TIMES = 1000 -> ~ 120 segundos
//// BLAS: Basic Linear Algebra Subroutines (MATLAB y NumPy)
//// cuBLAS for CUDA
//// LAPACK
//
//int main() {
//
//    float** A;
//    float* x;
//    float* b;
//    clock_t start, end;
//    double cpu_time_used;
//
//    // Allocate memory for A, x and b using malloc
//    A = (float**)malloc(ROWS * sizeof(float*));
//    for (int i = 0; i < ROWS; i++) {
//        A[i] = (float*)calloc(COLS, sizeof(float));
//    }
//    x = (float*)calloc(COLS, sizeof(float));
//    b = (float*)calloc(ROWS, sizeof(float));
//
//    // Initialize A and x with random values
//    srand(time(NULL));
//    for (int i = 0; i < ROWS; i++) {
//        for (int j = 0; j < COLS; j++) {
//            A[i][j] = (float)rand() / (float)(RAND_MAX / 10.0);
//        }
//        x[i] = (float)rand() / (float)(RAND_MAX / 10.0);
//    }
//
//
//    printf(" Comparing performance for dimension N = %d\n", ROWS);
//
//    // Perform A-x multiplication TIMES times and measure CPU time
//    start = clock();
//    for (int k = 0; k < TIMES; k++) {
//        for (int i = 0; i < ROWS; i++) {
//            b[i] = 0.0;
//            for (int j = 0; j < COLS; j++) {
//                b[i] += k * A[i][j] * x[j];
//
//            }
//
//            /*printf("%f10 ", b[i]);*/
//            //std::cout << std::endl;
//        }
//        //printf("%d ", k);
//    }
//    end = clock();
//    cpu_time_used = difftime(end, start) / CLOCKS_PER_SEC;
//    printf("CPU time used: %f seconds\n", cpu_time_used);
//}