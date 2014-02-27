/**
 *  Copyright (C) 2013 KU Leuven
 *
 *  This file is part of ApproxSVM.
 *
 *  ApproxSVM is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published
 *  by the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  ApproxSVM is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with ApproxSVM.  If not, see <http://www.gnu.org/licenses/>.
 *
 * MatrixMath.h
 *
 *      Author: Marc Claesen
 */

// In this file we implement all matrix-vector math that is used in EnsembleSVM.
// We interface to the BLAS when it is available and provide fallbacks in case it is not.
//
// Fallbacks are implemented via templates, which are replaced by specializations
// for double and float when the BLAS is available.
// By doing so, fallbacks are not compiled when the BLAS is available.

#ifndef MATRIXVECTOR_H_
#define MATRIXVECTOR_H_

#include "config.h"
#include <vector>
#include <cassert>
#include <functional>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <iostream>

using std::vector;
using std::size_t;

namespace approx{

/**
 * Class to model an iterator that strides over a vector.
 *
 * Iterating with stride s behaves equivalent to I+=s for a standard iterator I.
 */
template <typename T>
class stride_iter{
private:
	typename vector<T>::iterator I;
	size_t stride;

public:
	stride_iter(typename vector<T>::iterator I, size_t stride)
	:I(I),stride(stride){};
	stride_iter(const stride_iter &o)
	:I(o.I),stride(o.stride){};

	void operator=(stride_iter &o){
		I=o.I;
		stride=o.stride;
	}
	T &operator*() const{ return *I; };

	void operator++(){ I+=stride; }
	void operator--(){ I-=stride; }
	stride_iter operator+(size_t offset){ return stride_iter(I+stride*offset,stride); }
	stride_iter operator-(size_t offset){ return stride_iter(I-stride*offset,stride); }

	bool operator<(const stride_iter &rhs){ return I<rhs.I; }
	bool operator>(const stride_iter &rhs){ return I>rhs.I; }
	bool operator==(const stride_iter &rhs){ return (I==rhs.I); }
	bool operator!=(const stride_iter &rhs){ return !(operator==(rhs)); }

	bool operator<(const typename vector<T>::iterator &rhs){ return I<rhs; }
	bool operator>(const typename vector<T>::iterator &rhs){ return I>rhs; }
	bool operator==(const typename vector<T>::iterator &rhs){ return (I==rhs); }
	bool operator!=(const typename vector<T>::iterator &rhs){ return !(operator==(rhs)); }
	bool operator<(const typename vector<T>::const_iterator &rhs){ return I<rhs; }
	bool operator>(const typename vector<T>::const_iterator &rhs){ return I>rhs; }
	bool operator==(const typename vector<T>::const_iterator &rhs){ return (I==rhs); }
	bool operator!=(const typename vector<T>::const_iterator &rhs){ return !(operator==(rhs)); }
};

/**
 * Class to model a const_iterator that strides over a vector.
 *
 * Iterating with stride s behaves equivalent to I+=s for a standard iterator I.
 */
template <typename T>
class const_stride_iter{
private:
	typename vector<T>::const_iterator I;
	size_t stride;

public:
	const_stride_iter(typename vector<T>::const_iterator I, size_t stride)
	:I(I),stride(stride){};
	const_stride_iter(const const_stride_iter &o)
	:I(o.I),stride(o.stride){};

	void operator=(const_stride_iter &o){
		I=o.I;
		stride=o.stride;
	}
	const T &operator*() const{ return *I; };

	void operator++(){ I+=stride; }
	void operator--(){ I-=stride; }
	const_stride_iter operator+(size_t offset){ return const_stride_iter(I+stride*offset,stride); }
	const_stride_iter operator-(size_t offset){ return const_stride_iter(I-stride*offset,stride); }

	bool operator<(const const_stride_iter &rhs){ return I<rhs.I; }
	bool operator>(const const_stride_iter &rhs){ return I>rhs.I; }
	bool operator==(const const_stride_iter &rhs){ return (I==rhs.I); }
	bool operator!=(const const_stride_iter &rhs){ return !(operator==(rhs)); }

	bool operator<(const typename vector<T>::iterator &rhs){ return I<rhs; }
	bool operator>(const typename vector<T>::iterator &rhs){ return I>rhs; }
	bool operator==(const typename vector<T>::iterator &rhs){ return (I==rhs); }
	bool operator!=(const typename vector<T>::iterator &rhs){ return !(operator==(rhs)); }
	bool operator<(const typename vector<T>::const_iterator &rhs){ return I<rhs; }
	bool operator>(const typename vector<T>::const_iterator &rhs){ return I>rhs; }
	bool operator==(const typename vector<T>::const_iterator &rhs){ return (I==rhs); }
	bool operator!=(const typename vector<T>::const_iterator &rhs){ return !(operator==(rhs)); }
};

/**
 * Fallback implementation of dot-product if the BLAS is not available.
 */
template <typename T>
inline T dot(const std::vector<T> &x, const std::vector<T> &y){
	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
	return std::inner_product(x.begin(),x.end(),y.begin(),T(0));
}

/**
 *  GEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B )
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X**T,
 *
 *  alpha is a scalar, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  arows is the amount of rows of op(A)
 *
 *  Note: we assume Fortran-style matrices (column-major order).
 *
 *  Fallback implementation of alpha*A*B if the BLAS is not available.
 */
template <typename T>
inline void gemm(const vector<T> &A, bool atrans, const vector<T> &B, bool btrans, size_t arows, vector<T> &C, T alpha=1){
	size_t acols=A.size()/arows, brows=acols, bcols=B.size()/brows;
	C.resize(arows*bcols);

	typename vector<T>::const_iterator Ia=A.begin(), Ib=B.begin();
	if(atrans){
		if(btrans){
			// A and B transposed -- std for A, stride for B
			Ia=A.begin();
			for(size_t i=0;i<arows;++i){
				Ib=B.begin();
				for(size_t j=0;j<bcols;++j){
					const_stride_iter<T> ISb(Ib,bcols);
					C[i+j*arows]=alpha*std::inner_product(Ia,Ia+acols,ISb,T(0));

					// move Ib to next row
					Ib++;
				}
				// point Ia to next row
				Ia+=acols;
			}
		}else{
			// A transposed, B not transposed -- std for A and B
			Ia=A.begin();
			for(size_t i=0;i<arows;++i){
				Ib=B.begin();
				for(size_t j=0;j<bcols;++j){
					C[i+j*arows]=alpha*std::inner_product(Ia,Ia+acols,Ib,T(0));

					// move Ib to next column
					Ib+=brows;
				}
				// move Ia to next row
				Ia+=acols;
			}
		}
	}else{
		if(btrans){
			// A not transposed, B transposed -- stride for A and B
			Ia=A.begin();
			for(size_t i=0;i<arows;++i){
				const_stride_iter<T> ISa(Ia,arows);
				Ib=B.begin();
				for(size_t j=0;j<bcols;++j){
					const_stride_iter<T> ISb(Ib,bcols);
					C[i+j*arows]=alpha*std::inner_product(ISa,ISa+acols,ISb,T(0));

					// move Ib to next row
					Ib++;
				}
				// move Ia to next row
				Ia++;
			}
		}else{
			// A and B not transposed -- stride for A, std for B
			Ib=B.begin();
			for(size_t j=0;j<bcols;++j){ // B so move over cols
				Ia=A.begin();
				for(size_t i=0;i<arows;++i){  // A so move over rows
					const_stride_iter<T> ISa(Ia,arows);
					C[i+j*arows]=alpha*std::inner_product(ISa,ISa+acols,Ib,T(0));

					// move Ia to next row
					Ia++;
				}
				// move Ib to next column
				Ib+=brows;
			}
		}
	}
}

/**
 * Computes alpha*op(A)*D. D is a vector of diagonal elements.
 * Result is placed in A.
 *
 * op(A) = A if !atrans, otherwise op(A)=A^T
 *
 * Fastest if A is transposed.
 */
template <typename T>
inline void dmul(vector<T> &A, bool atrans, const vector<T> &D, T alpha=1){
	size_t acols=D.size(), arows=A.size()/acols;

	typename vector<T>::iterator I;
	if(atrans){
		// A is transposed, we need to stride
		I=A.begin();
		for(size_t i=0;i<arows;++i){
			stride_iter<T> IS(I,acols);
			std::transform(IS, IS+arows, IS, std::bind1st(std::multiplies<T>(),alpha*D[i]));
			I++;
		}
	}else{
		// A is not transposed
		I=A.begin();
		for(size_t i=0;i<acols;++i){
			std::transform(I, I+arows, I, std::bind1st(std::multiplies<T>(),alpha*D[i]));
			I+=arows;
		}
	}
}

/**
 * Computes the matrix-vector product alpha*op(M)*v.
 *
 * op(M) = trans ? M^T, M;
 *
 * Assumes Fortran-style matrix (column-major).
 *
 * Most efficient if M is transposed.
 */
template <typename T>
inline void gemv(const vector<T> &M, bool trans, const vector<T> &v, T alpha, vector<T> &result){
	size_t cols=v.size(), rows=M.size()/cols;
	result.clear();
	result.reserve(rows);

	if(trans){
		// M'*v
		typename vector<T>::const_iterator Im=M.begin(), Iv;
		for(size_t i=0;i<rows;++i){
			Iv=v.begin();
			result.push_back(alpha*std::inner_product(Im,Im+cols,Iv,T(0)));
			Im+=cols;
		}
	}else{
		// M*v
		typename vector<T>::const_iterator Im=M.begin(), Iv=v.begin();
		std::transform(Im,Im+rows,std::back_inserter(result),std::bind1st(std::multiplies<T>(),(*Iv)*alpha));
		Im+=rows;

		typename vector<T>::iterator Ir=result.begin();
		for(size_t i=1;i<cols;++i){
			Ir=result.begin();
			++Iv;
			while(Ir!=result.end())
				*(Ir++)+=alpha * (*(Im++)) * (*Iv);
		}
	}
}

/**
 * Computes the matrix-vector product result=alpha*M*v with M a symmetric matrix.
 */
template <typename T>
inline void symv(const vector<T> &M, const vector<T> &v, T alpha, vector<T> &result){
	size_t cols=v.size(), rows=M.size()/cols;
	result.clear();
	result.reserve(rows);

	typename vector<T>::const_iterator Im=M.begin(), Iv;
	for(size_t i=0;i<rows;++i){
		Iv=v.begin();
		result.push_back(alpha*std::inner_product(Im,Im+cols,Iv,T(0)));
		Im+=cols;
	}
}

/**
 * Computes the vector-matrix-vector product result=alpha*v^T*M*v with M a symmetric matrix.
 */
template <typename T>
inline float vtMv(const vector<T> &M, const vector<T> &v, T alpha=1){
	size_t cols=v.size();
	float result=0;

	typename vector<T>::const_iterator Im=M.begin(), Iv;
	for(size_t i=0;i<cols;++i){
		Iv=v.begin();
		result+=alpha*v[i]*std::inner_product(Im,Im+cols,Iv,T(0));
		Im+=cols;
	}
	return result;
}

/**
 * Helper function which prints M to os.
 */
template <typename T>
void printFortranMatrix(const std::vector<T> &M, size_t rows, bool transpose, std::ostream &os=std::cout){
	if(!transpose){
		size_t cols=M.size()/rows;
		typename std::vector<T>::const_iterator I=M.begin();
		for(size_t i=0;i<rows;++i){
			const_stride_iter<T> ISm(I,rows);

			// print
			for(size_t j=0;j<cols;++j,++ISm)
				os << *ISm << " ";
			os << std::endl;
			++I;
		}
	}else{
		size_t cols=M.size()/rows;
		for(size_t row=0;row<rows;++row){
			for(size_t col=0;col<cols;++col)
				os << M[row*cols+col] << " ";
			os << std::endl;
		}
	}
}

// start of BLAS interfaces when it is available
#if defined(HAVE_LIBBLAS) || defined(HAVE_LIBATLAS)

/**
 * Interface to the BLAS SDOT function.
 * Returns the inner product of x and y.
 */
//float dot(const vector<float> &x, const vector<float> &y);
// fallback is faster with a compiler that outputs SIMD instructions (e.g. gcc/intel)
// or is it? fixme

/**
 * Interface to the BLAS DDOT function.
 * Returns the inner product of x and y.
 */
//double dot(const vector<double> &x, const vector<double> &y);
// fallback is faster with a compiler that outputs SIMD instructions (e.g. gcc/intel)
// or is it? fixme

/**
 *  GEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B )
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X**T,
 *
 *  alpha is a scalar, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  Note: we assume Fortran-style matrices (column-major order).
 *
 * Interface to the BLAS SGEMM function.
 */
void gemm(const vector<float> &A, bool atrans, const vector<float> &B, bool btrans, size_t arows, vector<float> &C, float alpha=1);

/**
 *  GEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B )
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X**T,
 *
 *  alpha is a scalar, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  Note: we assume Fortran-style matrices (column-major order).
 *
 * Interface to the BLAS DGEMM function.
 */
void gemm(const vector<double> &A, bool atrans, const vector<double> &B, bool btrans, size_t arows, vector<double> &C, double alpha=1);

/**
 * Computes the matrix-vector product alpha*op(M)*v.
 *
 * op(M) = M if !trans, otherwise op(M)=M^T
 *
 * Interface to BLAS SGEMV.
 */
void gemv(const vector<float> &M, bool trans, const vector<float> &v, float alpha, vector<float> &result);

/**
 * Computes the matrix-vector product alpha*op(M)*v.
 *
 * op(M) = M if !trans, otherwise op(M)=M^T
 *
 * Interface to BLAS DGEMV.
 */
void gemv(const vector<double> &M, bool trans, const vector<double> &v, double alpha, vector<double> &result);

/**
 * Computes the matrix-vector product result=alpha*M*v with M a symmetric matrix.
 *
 * Interface to BLAS SSYMV.
 */
void symv(const vector<float> &M, const vector<float> &v, float alpha, vector<float> &result);

/**
 * Computes the matrix-vector product result=alpha*M*v with M a symmetric matrix.
 *
 * Interface to BLAS DSYMV.
 */
void symv(const vector<double> &M, const vector<double> &v, double alpha, vector<double> &result);

#endif // end of BLAS interfaces

} // approx namespace

#endif /* MATRIXVECTOR_H_ */
