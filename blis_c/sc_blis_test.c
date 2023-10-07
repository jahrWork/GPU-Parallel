#include <stdio.h>
#include <time.h>
#include <math.h>
#include <unistd.h> // Used for pausing to let the laptop cool down
#include "blis.h"

void print_array(const float integers[], int elements) {

  // This function prints the results as an array to use for plots
  const char *separator = ""; 
  for(int j=0; j<elements; j++) {
    printf("%s %f", separator, integers[j]);
    separator = ",";
  }
  printf("\n");
}

int main(int argc, char **argv)
{
  // Main parameters to use
  num_t dt_r, dt_c;
  dim_t n;
  inc_t rs, cs;
  int64_t large_n, TIMES, i;
  int j;

  float results[161];

  obj_t a, b, c;
  obj_t *alpha;
  obj_t *beta;

  large_n = 10000;
  printf("\n#\n#  -- Example 1 --\n#\n\n");

  // Create some matrix operands to work with.
  dt_r = BLIS_SINGLE;
  dt_c = BLIS_SINGLE;
  rs = 0;
  cs = 0;
  
  // Set the scalars to use.
  alpha = &BLIS_ONE;
  beta = &BLIS_ZERO;
  
  j = 0;
  for (n = 50; n <= 2500; n += 25){

    TIMES = (large_n*large_n*large_n) / (n*n*n);
    printf("N = %ld; TIMES = %ld \n", n, TIMES);

    // Create the matrices to operate.
    bli_obj_create(dt_c, n, n, rs, cs, &c);
    bli_obj_create(dt_r, n, n, rs, cs, &a);
    bli_obj_create(dt_c, n, n, rs, cs, &b);

    // Initialize the matrix operands.
    bli_randm(&a);
    bli_randm(&b);
    bli_setm(&BLIS_ZERO, &c);

    clock_t start = clock();
    for (i = 0; i < TIMES; ++i){
      bli_gemm(alpha, &a, &b, beta, &c);
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    results[j] = seconds;
    printf("Time taken: %10.3f seconds \n \n", seconds);

    // Comment the next line to run the program without pauses
    sleep(10);

    bli_obj_free(&a);
    bli_obj_free(&b);
    bli_obj_free(&c);

    j += 1;
  }

  for (TIMES = 63; TIMES > 0; TIMES--){

    n = pow((large_n*large_n*large_n) / TIMES, 1./3);
    printf("N = %ld; TIMES = %ld \n", n, TIMES);

    bli_obj_create(dt_c, n, n, rs, cs, &c);
    bli_obj_create(dt_r, n, n, rs, cs, &a);
    bli_obj_create(dt_c, n, n, rs, cs, &b);

    // Initialize the matrix operands.
    bli_randm(&a);
    bli_randm(&b);
    bli_setm(&BLIS_ZERO, &c);

    clock_t start = clock();
    for (i = 0; i < TIMES; ++i){
      bli_gemm(alpha, &a, &b, beta, &c);
    }
    clock_t end = clock();
    float seconds = (float)(end - start) / CLOCKS_PER_SEC;
    results[j] = seconds;
    printf("Time taken: %10.3f seconds \n \n", seconds);

    // Comment the next line to run the program without pauses
    sleep(10);

    bli_obj_free(&a);
    bli_obj_free(&b);
    bli_obj_free(&c);

    j += 1;
  }

  print_array(results, sizeof(results)/sizeof(results[0]));

  return 0;
}
