// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "acsmatmult/matmult.h"
#include <immintrin.h>  // Intel intrinsics for SSE/AVX.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

typedef union _avxd {
  __m256d val;
  double arr[4];
} avxd;

Matrix<float> multiplyMatricesSIMD(Matrix<float> a, Matrix<float> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  auto result = Matrix<float>(a.rows, b.columns); 
  
    __m256 va, vb, vtemp;
    __m128 vlow, vhigh, vresult;


    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.columns; j++) {
            for (int k = 0; k < a.columns; k += 8) {
                // load
                // va = _mm256_loadu_ps(a+(i*1024)+k); // matrix_a[i][k]
                va = _mm256_loadu_ps((float*)&a(i, k));
                // vb = _mm256_loadu_ps(a+(j*1024)+k); // matrix_b[j][k]
                vb = _mm256_loadu_ps((float*)&b(i, k));

                // multiply
                vtemp = _mm256_mul_ps(va, vb);

                // add
                // extract higher four floats
                vhigh = _mm256_extractf128_ps(vtemp, 1); // high 128
                // add higher four floats to lower floats
                vresult = _mm_add_ps(_mm256_castps256_ps128(vtemp), vhigh);
                // horizontal add of that result
                vresult = _mm_hadd_ps(vresult, vresult);
                // another horizontal add of that result
                vresult = _mm_hadd_ps(vresult, vresult);

                // store
                result(i,j) += _mm_cvtss_f32(vresult);
            }
        }
    }
    return result;
}


Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
      auto result = Matrix<double>(a.rows, b.columns); 
  
    __m256 va, vb, vtemp;
    __m128 vlow, vhigh, vresult;


    for (int i = 0; i < a.rows; i++) {
        for (int j = 0; j < b.columns; j++) {
            for (int k = 0; k < a.columns; k += 8) {
                // load
                // va = _mm256_loadu_ps(a+(i*1024)+k); // matrix_a[i][k]
                __m256d va = _mm256_loadu_pd((double*)&a(i, k));
                // vb = _mm256_loadu_ps(a+(j*1024)+k); // matrix_b[j][k]
                __m256d vb = _mm256_loadu_pd((double*)&b(i, k));

                // multiply
                __m256d vtemp = _mm256_mul_pd(va, vb);

                // add
                // extract higher four floats
                __m128d vhigh = _mm256_extractf128_pd(vtemp, 1); // high 128
                // add higher four floats to lower floats
                __m128d vresult = _mm_add_pd(_mm256_castpd256_pd128(vtemp), vhigh);
                // horizontal add of that result
                 vresult = _mm_hadd_pd(vresult, vresult);
                // another horizontal add of that result
                 vresult = _mm_hadd_pd(vresult, vresult);

                // store
                // result(i,j) += _mm_cvtss_si32(vresult);
                _mm_storeu_pd((double*)&result(i, j) , vresult);
            }
        }
    }
    return result;
}

/*************************************/
#pragma GCC pop_options
/*************************************/
