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
    if (a.columns != b.rows) {
      throw std::domain_error("Matrix dimensions do not allow matrix-multiplication.");
    }

    // Height and width of resulting matrix
    auto res_rows = a.rows;
    auto res_columns = b.columns;
    auto mul_range = a.columns;
    auto result = Matrix<float>(res_rows, res_columns);
    
    for(auto i = 0; i < res_rows; i++)
	    for(auto j = 0; j < res_columns; j++) //Iter over rows of Right matrix
	    {
        __m256 a_ij = _mm256_set1_ps(a(i, j));
          int k = 0;

        while(k < mul_range) {
          if ((mul_range - k) >= 32) {
            //std::cout << "Entered if mul_range - k >= 32 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256 b_jk0 =  _mm256_loadu_ps((float*)&b(j, k));
            __m256 b_jk8 =  _mm256_loadu_ps((float*)&b(j, k+8));
            __m256 b_jk16 =  _mm256_loadu_ps((float*)&b(j, k+16));
            __m256 b_jk24 =  _mm256_loadu_ps((float*)&b(j, k+24)); 

            //Load 32 elements from the ith row of result matrix
            __m256 r_ik0 ;
            __m256 r_ik8 ;
            __m256 r_ik16;
            __m256 r_ik24;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_ps();
            r_ik8  = _mm256_setzero_ps();
            r_ik16 = _mm256_setzero_ps();
            r_ik24 = _mm256_setzero_ps();
            }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_ps((float*)&result(i, k));
            r_ik8  = _mm256_loadu_ps((float*)&result(i, k+8));
            r_ik16 = _mm256_loadu_ps((float*)&result(i, k+16));
            r_ik24 = _mm256_loadu_ps((float*)&result(i, k+24));
            }
            

            //Mutiply all 32 elements of right matrix with a_ij element of left matrix
            __m256 mul0 = _mm256_mul_ps(a_ij, b_jk0);
            __m256 mul1 = _mm256_mul_ps(a_ij, b_jk8);
            __m256 mul2 = _mm256_mul_ps(a_ij, b_jk16);
            __m256 mul3 = _mm256_mul_ps(a_ij, b_jk24);

            //Add the multiplication with result vector
            r_ik0  = _mm256_add_ps(r_ik0 , mul0);
            r_ik8  = _mm256_add_ps(r_ik8 , mul1);
            r_ik16 = _mm256_add_ps(r_ik16, mul2);
            r_ik24 = _mm256_add_ps(r_ik24, mul3);

            //Store mul result in memory
            _mm256_storeu_ps((float*)&result(i, k)   , r_ik0 );
            _mm256_storeu_ps((float*)&result(i, k+8) , r_ik8 );
            _mm256_storeu_ps((float*)&result(i, k+16), r_ik16);
            _mm256_storeu_ps((float*)&result(i, k+24), r_ik24);

            //Add 32 to k
            k += 32;
          }
          else if(mul_range - k >= 16) {
            //std::cout << "Entered if mul_range - k >= 16 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256 b_jk0 =  _mm256_loadu_ps((float*)&b(j, k));
            __m256 b_jk8 =  _mm256_loadu_ps((float*)&b(j, k+8));

            //Load 32 elements from the ith row of result matrix
            __m256 r_ik0 ;
            __m256 r_ik8 ;
           
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_ps();
            r_ik8  = _mm256_setzero_ps();
           }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_ps((float*)&result(i, k));
            r_ik8  = _mm256_loadu_ps((float*)&result(i, k+8));
            
            }
            
            //Mutiply all 32 elements of right matrix with a_ij element of left matrix
            __m256 mul0 = _mm256_mul_ps(a_ij, b_jk0);
            __m256 mul1 = _mm256_mul_ps(a_ij, b_jk8);

            //Add the multiplication with result vector
            r_ik0  = _mm256_add_ps(r_ik0 , mul0);
            r_ik8  = _mm256_add_ps(r_ik8 , mul1);

            //Store mul result in memory
            _mm256_storeu_ps((float*)&result(i, k)   , r_ik0 );
            _mm256_storeu_ps((float*)&result(i, k+8) , r_ik8 );

            //Add 16 to k
            k += 16;
          }
          else if(mul_range - k >= 8) {
            //std::cout << "Entered if mul_range - k >= 8 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256 b_jk0 =  _mm256_loadu_ps((float*)&b(j, k));
            //Load 32 elements from the ith row of result matrix
            __m256 r_ik0 ;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_ps();
            }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_ps((float*)&result(i, k));
            }
            //Mutiply all 32 elements of right matrix with a_ij element of left matrix
            __m256 mul0 = _mm256_mul_ps(a_ij, b_jk0);
            //Add the multiplication with result vector
            r_ik0  = _mm256_add_ps(r_ik0 , mul0);
            //Store mul result in memory
            _mm256_storeu_ps((float*)&result(i, k)   , r_ik0 );
            //Add 32 to k
            k += 8;
          }
          else if(mul_range - k >= 4) {
            //std::cout << "Entered if mul_range - k >= 16 loop" << std::endl;
            __m128 a_ij = _mm_set1_ps(a(i, j));
            //Load 4 elements from the jth row of right matrix 
            __m128 b_jk0 =  _mm_loadu_ps((float*)&b(j, k));
            //Load 4 elements from the ith row of result matrix
            __m128 r_ik0 ;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm_setzero_ps();
            }
            else { //Load already initialized results
            r_ik0  = _mm_loadu_ps((float*)&result(i, k));    
            }
            //Mutiply all 4 elements of right matrix with a_ij element of left matrix
            __m128 mul0 = _mm_mul_ps(a_ij, b_jk0);
            //Add the multiplication with result vector
            r_ik0  = _mm_add_ps(r_ik0 , mul0);       
            //Store mul result in memory
            _mm_storeu_ps((float*)&result(i, k)   , r_ik0 );
            //Add 4 to k
            k += 4;
          }
          else {
            // std::cout << "Entered else" << std::endl;
            //float aij = a(i, j);
            result(i,k) += a(i,j)*b(j,k);
            k++;
          }
        }
	    }
  return result;
}

Matrix<double> multiplyMatricesSIMD(Matrix<double> a,
                                  Matrix<double> b) {
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
  /* REPLACE THE CODE IN THIS FUNCTION WITH YOUR OWN CODE */
  /* YOU MUST USE VECTOR EXTENSIONS HERE */
    if (a.columns != b.rows) {
      throw std::domain_error("Matrix dimensions do not allow matrix-multiplication.");
    }

    // Height and width of resulting matrix
    auto res_rows = a.rows;
    auto res_columns = b.columns;
    auto mul_range = a.columns;
    auto result = Matrix<double>(res_rows, res_columns);
    
    for(auto i = 0; i < res_rows; i++)
	    for(auto j = 0; j < res_columns; j++) //Iter over rows of Right matrix
	    {
        __m256d a_ij = _mm256_set1_pd(a(i, j));
          int k = 0;

        while(k < mul_range) {
          if ((mul_range - k) >= 16) {
            //std::cout << "Entered if mul_range - k >= 32 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256d b_jk0 =   _mm256_loadu_pd((double*)&b(j, k));
            __m256d b_jk4 =   _mm256_loadu_pd((double*)&b(j, k+4));
            __m256d b_jk8 =  _mm256_loadu_pd((double*)&b(j, k+8));
            __m256d b_jk12 =  _mm256_loadu_pd((double*)&b(j, k+12)); 

            //Load 32 elements from the ith row of result matrix
            __m256d r_ik0 ;
            __m256d r_ik4 ;
            __m256d r_ik8 ;
            __m256d r_ik12;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_pd();
            r_ik4  = _mm256_setzero_pd();
            r_ik8  = _mm256_setzero_pd();
            r_ik12 = _mm256_setzero_pd();
            }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_pd((double*)&result(i, k));
            r_ik4  = _mm256_loadu_pd((double*)&result(i, k+4));
            r_ik8  = _mm256_loadu_pd((double*)&result(i, k+8));
            r_ik12 = _mm256_loadu_pd((double*)&result(i, k+12));
            }
            

            //Mutiply all 32 elements of right matrix with a_ij element of left matrix
            __m256d mul0 = _mm256_mul_pd(a_ij, b_jk0);
            __m256d mul1 = _mm256_mul_pd(a_ij, b_jk4);
            __m256d mul2 = _mm256_mul_pd(a_ij, b_jk8);
            __m256d mul3 = _mm256_mul_pd(a_ij, b_jk12);

            //Add the multiplication with result vector
            r_ik0  = _mm256_add_pd(r_ik0 , mul0);
            r_ik4  = _mm256_add_pd(r_ik4 , mul1);
            r_ik8  = _mm256_add_pd(r_ik8 , mul2);
            r_ik12 = _mm256_add_pd(r_ik12, mul3);

            //Store mul result in memory
            _mm256_storeu_pd((double*)&result(i, k)   , r_ik0 );
            _mm256_storeu_pd((double*)&result(i, k+4) , r_ik4 );
            _mm256_storeu_pd((double*)&result(i, k+8) , r_ik8 );
            _mm256_storeu_pd((double*)&result(i, k+12), r_ik12);

            //Add 32 to k
            k += 16;
          }
          else if(mul_range - k >= 8) {
            //std::cout << "Entered if mul_range - k >= 16 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256d b_jk0 =  _mm256_loadu_pd((double*)&b(j, k));
            __m256d b_jk4 =  _mm256_loadu_pd((double*)&b(j, k+4));

            //Load 32 elements from the ith row of result matrix
            __m256d r_ik0 ;
            __m256d r_ik4 ;
           
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_pd();
            r_ik4  = _mm256_setzero_pd();
           }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_pd((double*)&result(i, k));
            r_ik4  = _mm256_loadu_pd((double*)&result(i, k+4));
            }
            
            //Mutiply all 8 elements of right matrix with a_ij element of left matrix
            __m256d mul0 = _mm256_mul_pd(a_ij, b_jk0);
            __m256d mul1 = _mm256_mul_pd(a_ij, b_jk4);

            //Add the multiplication with result vector
            r_ik0  = _mm256_add_pd(r_ik0 , mul0);
            r_ik4  = _mm256_add_pd(r_ik4 , mul1);

            //Store mul result in memory
            _mm256_storeu_pd((double*)&result(i, k)   , r_ik0 );
            _mm256_storeu_pd((double*)&result(i, k+4) , r_ik4 );

            //Add 8 to k
            k += 8;
          }
          else if(mul_range - k >= 4) {
            //std::cout << "Entered if mul_range - k >= 8 loop" << std::endl;
            //Load 32 elements from the jth row of right matrix 
            __m256d b_jk0 =  _mm256_loadu_pd((double*)&b(j, k));
            //Load 32 elements from the ith row of result matrix
            __m256d r_ik0 ;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm256_setzero_pd();
            }
            else { //Load already initialized results
            r_ik0  = _mm256_loadu_pd((double*)&result(i, k));
            }
            //Mutiply all 32 elements of right matrix with a_ij element of left matrix
            __m256d mul0 = _mm256_mul_pd(a_ij, b_jk0);
            //Add the multiplication with result vector
            r_ik0  = _mm256_add_pd(r_ik0 , mul0);
            //Store mul result in memory
            _mm256_storeu_pd((double*)&result(i, k)   , r_ik0 );
            //Add 32 to k
            k += 4;
          }
          else if(mul_range - k >= 2) {
            //std::cout << "Entered if mul_range - k >= 16 loop" << std::endl;
            __m128d a_ij = _mm_set1_pd(a(i, j));
            //Load 4 elements from the jth row of right matrix 
            __m128d b_jk0 =  _mm_loadu_pd((double*)&b(j, k));
            //Load 4 elements from the ith row of result matrix
            __m128d r_ik0 ;
            if (i==0 && j==0) { //Initialize results
            r_ik0  = _mm_setzero_pd();
            }
            else { //Load already initialized results
            r_ik0  = _mm_loadu_pd((double*)&result(i, k));    
            }
            //Mutiply all 4 elements of right matrix with a_ij element of left matrix
            __m128d mul0 = _mm_mul_pd(a_ij, b_jk0);
            //Add the multiplication with result vector
            r_ik0  = _mm_add_pd(r_ik0 , mul0);       
            //Store mul result in memory
            _mm_storeu_pd((double*)&result(i, k)   , r_ik0 );
            //Add 4 to k
            k += 2;
          }
          else {
            //std::cout << "Entered else" << std::endl;
            //float aij = a(i, j);
            result(i,k) = result(i,k) + a(i,j)*b(j,k);
            k++;
          }
        }
	    }
  return result;
  //return Matrix<double>(1, 1);
}

/*************************************/
#pragma GCC pop_options
/*************************************/
