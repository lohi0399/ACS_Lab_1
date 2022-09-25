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
#include <omp.h>  // OpenMP support.

/* You may not remove these pragmas: */
/*************************************/
#pragma GCC push_options
#pragma GCC optimize ("O0")
/*************************************/

Matrix<float> multiplyMatricesOMP(Matrix<float> a,
                                  Matrix<float> b,
                                  int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  auto c = Matrix<float>(a.rows,b.columns);
  size_t i,j,k;

#pragma omp parallel for private(i,j,k) shared(a,b,c)
  for ( i = 0; i < a.rows; i++) {
      for ( j = 0; j < b.columns; j++) {
          for ( k = 0; k < a.rows; k++) {
              c(i,j) += a(i,k) * b(k,j);
          }
      }
  }

  /* YOU MUST USE OPENMP HERE */
  return c;
}

Matrix<double> multiplyMatricesOMP(Matrix<double> a,
                                   Matrix<double> b,
                                   int num_threads) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  auto c = Matrix<double>(a.rows,b.columns);
  size_t i,j,k;

#pragma omp parallel for private(i,j,k) shared(a,b,c)
  for ( i = 0; i < a.rows; i++) {
      for ( j = 0; j < b.columns; j++) {
          for ( k = 0; k < a.rows; k++) {
              c(i,j) += a(i,k) * b(k,j);
          }
      }
  }

  /* YOU MUST USE OPENMP HERE */
  return c;
}
#pragma GCC pop_options
