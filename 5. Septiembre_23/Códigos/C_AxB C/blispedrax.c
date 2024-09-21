#include <stdio.h>
#include <time.h>
#include "blis.h"


#include <stdio.h>
#include <time.h>
#include "blis.h"

int main(int argc, char** argv)
{
    num_t dt_r, dt_c;
    dim_t m, n, k;
    inc_t rs, cs;
    long long int N_ops, TIMES;
    double elapsed_time;

    obj_t a, b, c;
    obj_t alpha, beta;

    dt_r = BLIS_SINGLE_PREC;
    dt_c = BLIS_SINGLE_PREC;
    rs = 0; cs = 0;

    bli_obj_create(BLIS_FLOAT, 1, 1, 0, 0, &alpha);
    bli_setsc(1.0, 0.0, &alpha);

    bli_obj_create(BLIS_FLOAT, 1, 1, 0, 0, &beta);
    bli_setsc(0.0, 0.0, &beta);

    N_ops = 2LL * 10000 * 10000 * 10000;

    printf("Inicio de la prueba hola\n");

    dim_t N_values[14] = {4149, 4253, 4368, 4496, 4642, 4807, 5000, 5228, 5503, 5848, 6300, 6934, 7937, 10000};

    long long int TIMES_values[14] = {14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};

    for (int i = 0; i < 14; i++)
    {
        m = n = k = N_values[i];
        TIMES = TIMES_values[i];

        bli_obj_create(dt_c, m, n, rs, cs, &c);
        bli_obj_create(dt_r, m, k, rs, cs, &a);
        bli_obj_create(dt_c, k, n, rs, cs, &b);

        bli_randm(&a);
        bli_randm(&b);
        bli_setm(&beta, &c);

        clock_t start = clock();

        for (long long int j = 0; j < TIMES; j++)
        {
            bli_gemm(&alpha, &a, &b, &beta, &c);
        }

        clock_t end = clock();
        elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

        printf("N: %ld, TIMES: %lld, Execution Time: %f seconds\n", N_values[i], TIMES, elapsed_time);

        bli_obj_free(&a);
        bli_obj_free(&b);
        bli_obj_free(&c);
    }

    bli_obj_free(&alpha);
    bli_obj_free(&beta);

    return 0;
}


// int main( int argc, char** argv )
// {
//     num_t dt_r, dt_c;
//     dim_t m, n, k;
//     inc_t rs, cs;
//     long long int N_ops, TIMES;
//     double elapsed_time;

//     obj_t a, b, c;
//     obj_t* alpha;
//     obj_t* beta;

//     dt_r = BLIS_SINGLE_PREC;
//     dt_c = BLIS_SINGLE_PREC;
//     rs = 0; cs = 0;
//     alpha = &BLIS_ONE;
//     beta  = &BLIS_ZERO;
//     N_ops = 2 * 10000LL * 10000LL * 10000LL;

//     printf("Inicio de la prueba hola\n");


//     for (dim_t dim = 20; dim <= 4000; dim += 20)
//     {
//         m = n = k = dim;
//         TIMES = N_ops / (2LL * m * n * k);
        
//         bli_obj_create( dt_c, m, n, rs, cs, &c );
//         bli_obj_create( dt_r, m, k, rs, cs, &a );
//         bli_obj_create( dt_c, k, n, rs, cs, &b );

//         bli_randm( &a );
//         bli_randm( &b );
//         bli_setm( &BLIS_ZERO, &c );

//         clock_t start = clock();

//         for (long long int i = 0; i < TIMES; i++)
//         {
//             bli_gemm( alpha, &a, &b, beta, &c );
//         }

//         clock_t end = clock();
//         elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;

//         printf("N: %ld, TIMES: %lld, Execution Time: %f seconds\n", dim, TIMES, elapsed_time);

//         bli_obj_free( &a );
//         bli_obj_free( &b );
//         bli_obj_free( &c );
//     }

//     return 0;
// }
