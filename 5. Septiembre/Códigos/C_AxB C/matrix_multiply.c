/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    - Neither the name of The University of Texas nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <stdio.h>
#include "blis.h"

int main(int argc, char **argv)
{
  num_t dt_r, dt_c;
  num_t dt_s, dt_d;
  num_t dt_a, dt_b;
  dim_t m, n, k;
  inc_t rs, cs;

  obj_t a, b, c;
  obj_t *alpha;
  obj_t *beta;

  //
  // This file demonstrates mixing datatypes in gemm.
  //
  // NOTE: Please make sure that mixed datatype support is enabl"-I${workspaceFolder}/../..",ed in BLIS
  // before proceeding to build and run the example binaries. If you're not
  // sure whether mixed datatype support is enabled in BLIS, please refer
  // to './configure --help' for the relevant options.
  //

  //
  // Example 1: Perform a general matrix-matrix multiply (gemm) operation
  //            with operands of different domains (but identical precisions).
  //

  printf("\n#\n#  -- Example 1 --\n#\n\n");

  // Create some matrix operands to work with.
  dt_r = BLIS_DOUBLE;
  dt_c = BLIS_DOUBLE;
  m = 4;
  n = 5;
  k = 1;
  rs = 0;
  cs = 0;
  bli_obj_create(dt_c, m, n, rs, cs, &c);
  bli_obj_create(dt_r, m, k, rs, cs, &a);
  bli_obj_create(dt_c, k, n, rs, cs, &b);

  // Set the scalars to use.
  alpha = &BLIS_ONE;
  beta = &BLIS_ONE;

  // Initialize the matrix operands.
  bli_randm(&a);
  bli_randm(&b);
  bli_setm(&BLIS_ZERO, &c);

  bli_printm("a (double real):    randomized", &a, "%4.1f", "");
  bli_printm("b (double complex): randomized", &b, "%4.1f", "");
  bli_printm("c (double complex): initial value", &c, "%4.1f", "");

  // c := beta * c + alpha * a * b, where 'a' is real, and 'b' and 'c' are
  // complex.
  bli_gemm(alpha, &a, &b, beta, &c);

  bli_printm("c (double complex): after gemm", &c, "%4.1f", "");

  // Free the objects.
  bli_obj_free(&a);
  bli_obj_free(&b);
  bli_obj_free(&c);

  return 0;
}
