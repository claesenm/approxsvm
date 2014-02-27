/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * MatrixMath.cpp
 *
 *      Author: Marc Claesen
 */

#include "MatrixMath.h"

#ifdef HAVE_LIBATLAS

namespace{

#ifndef CBLAS_ENUM_DEFINED_H
   #define CBLAS_ENUM_DEFINED_H
   enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102 };
   enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113,
                         AtlasConj=114};
   enum CBLAS_UPLO  {CblasUpper=121, CblasLower=122};
   enum CBLAS_DIAG  {CblasNonUnit=131, CblasUnit=132};
   enum CBLAS_SIDE  {CblasLeft=141, CblasRight=142};
#endif

//#ifndef CBLAS_ENUM_DEFINED_H
//#define CBLAS_ENUM_DEFINED_H
//struct CBLAS_ENUMS{
//	enum CBLAS_ORDER { CblasRowMajor=101, CblasColMajor=102 };
//	enum CBLAS_TRANSPOSE { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114 };
//	enum CBLAS_UPLO{ CblasUpper=121, CblasLower=122 };
//	enum CBLAS_DIAG { CblasNonUnit=131, CblasUnit=132 };
//	enum CBLAS_SIDE { CblasLeft=141, CblasRight=142 };
//};
//#endif

//struct CBLAS_ORDER{
//	enum { CblasRowMajor=101, CblasColMajor=102 };
//};
//struct CBLAS_TRANSPOSE{
//	enum { CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113, AtlasConj=114 };
//};
//struct CBLAS_UPLO{
//	enum {CblasUpper=121, CblasLower=122};
//};
//struct CBLAS_DIAG{
//	enum { CblasNonUnit=131, CblasUnit=132 };
//};
//struct CBLAS_SIDE{
//	enum { CblasLeft=141, CblasRight=142 };
//};

extern "C"{

//float cblas_sdot(const int N, const float  *X, const int incX,
//                  const float  *Y, const int incY);

void cblas_sgemv(const enum CBLAS_ORDER Order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const float alpha, const float *A, const int lda,
                 const float *X, const int incX, const float beta,
                 float *Y, const int incY);

void cblas_sgemm(const enum CBLAS_ORDER Order, const enum CBLAS_TRANSPOSE TransA,
                 const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A,
                 const int lda, const float *B, const int ldb,
                 const float beta, float *C, const int ldc);

void cblas_ssymv(const enum CBLAS_ORDER Order, const enum CBLAS_UPLO Uplo,
                 const int N, const float alpha, const float *A,
                 const int lda, const float *X, const int incX,
                 const float beta, float *Y, const int incY);

} // extern C

} // anonymous namespace

#else
#ifdef HAVE_LIBBLAS

// This file should only be included for compilation when BLAS is available.
// If BLAS is found by the configure script, the CPP macro HAVE_BLAS is defined.

namespace{

extern "C"{

typedef char character;
typedef float real;
typedef double doublereal;
typedef long int integer;

//real sdot_(integer *n, real *x, integer *incx, real *y, integer *incy);
//doublereal ddot_(integer *n, doublereal *x, integer *incx, doublereal *y, integer *incy);

/**
 * SUBROUTINE SSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
 *
 *  Purpose
 *  =======
 *
 *  SSYMV  performs the matrix-vector  operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the array A is to be referenced as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - REAL            .
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - REAL             array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular part of the symmetric matrix and the strictly
 *           lower triangular part of A is not referenced.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular part of the symmetric matrix and the strictly
 *           upper triangular part of A is not referenced.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *  X      - REAL             array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - REAL            .
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - REAL             array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y. On exit, Y is overwritten by the updated
 *           vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  Further Details
 *  ===============
 *
 *  Level 2 Blas routine.
 *  The vector and matrix arguments are not referenced when N = 0, or M = 0
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *  =====================================================================
 */
void ssymv_(character *UPLO, integer *N, real *ALPHA, real *A, integer *LDA, real *X, integer *INCX, real *BETA, real *Y, integer *incy);
void dsymv_(character *UPLO, integer *N, doublereal *ALPHA, doublereal *A, integer *LDA, doublereal *X, integer *INCX, doublereal *BETA, doublereal *Y, integer *incy);

/**
      SUBROUTINE DGEMM(TRANSA,TRANSB,M,N,K,ALPHA,A,LDA,B,LDB,BETA,C,LDC)
 *     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER K,LDA,LDB,LDC,M,N
      CHARACTER TRANSA,TRANSB
 *     ..
 *     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),B(LDB,*),C(LDC,*)
 *     ..
 *
 *  Purpose
 *  =======
 *
 *  DGEMM  performs one of the matrix-matrix operations
 *
 *     C := alpha*op( A )*op( B ) + beta*C,
 *
 *  where  op( X ) is one of
 *
 *     op( X ) = X   or   op( X ) = X**T,
 *
 *  alpha and beta are scalars, and A, B and C are matrices, with op( A )
 *  an m by k matrix,  op( B )  a  k by n matrix and  C an m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  TRANSA - CHARACTER*1.
 *           On entry, TRANSA specifies the form of op( A ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSA = 'N' or 'n',  op( A ) = A.
 *
 *              TRANSA = 'T' or 't',  op( A ) = A**T.
 *
 *              TRANSA = 'C' or 'c',  op( A ) = A**T.
 *
 *           Unchanged on exit.
 *
 *  TRANSB - CHARACTER*1.
 *           On entry, TRANSB specifies the form of op( B ) to be used in
 *           the matrix multiplication as follows:
 *
 *              TRANSB = 'N' or 'n',  op( B ) = B.
 *
 *              TRANSB = 'T' or 't',  op( B ) = B**T.
 *
 *              TRANSB = 'C' or 'c',  op( B ) = B**T.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry,  M  specifies  the number  of rows  of the  matrix
 *           op( A )  and of the  matrix  C.  M  must  be at least  zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry,  N  specifies the number  of columns of the matrix
 *           op( B ) and the number of columns of the matrix C. N must be
 *           at least zero.
 *           Unchanged on exit.
 *
 *  K      - INTEGER.
 *           On entry,  K  specifies  the number of columns of the matrix
 *           op( A ) and the number of rows of the matrix op( B ). K must
 *           be at least  zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, ka ), where ka is
 *           k  when  TRANSA = 'N' or 'n',  and is  m  otherwise.
 *           Before entry with  TRANSA = 'N' or 'n',  the leading  m by k
 *           part of the array  A  must contain the matrix  A,  otherwise
 *           the leading  k by m  part of the array  A  must contain  the
 *           matrix A.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. When  TRANSA = 'N' or 'n' then
 *           LDA must be at least  max( 1, m ), otherwise  LDA must be at
 *           least  max( 1, k ).
 *           Unchanged on exit.
 *
 *  B      - DOUBLE PRECISION array of DIMENSION ( LDB, kb ), where kb is
 *           n  when  TRANSB = 'N' or 'n',  and is  k  otherwise.
 *           Before entry with  TRANSB = 'N' or 'n',  the leading  k by n
 *           part of the array  B  must contain the matrix  B,  otherwise
 *           the leading  n by k  part of the array  B  must contain  the
 *           matrix B.
 *           Unchanged on exit.
 *
 *  LDB    - INTEGER.
 *           On entry, LDB specifies the first dimension of B as declared
 *           in the calling (sub) program. When  TRANSB = 'N' or 'n' then
 *           LDB must be at least  max( 1, k ), otherwise  LDB must be at
 *           least  max( 1, n ).
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry,  BETA  specifies the scalar  beta.  When  BETA  is
 *           supplied as zero then C need not be set on input.
 *           Unchanged on exit.
 *
 *  C      - DOUBLE PRECISION array of DIMENSION ( LDC, n ).
 *           Before entry, the leading  m by n  part of the array  C must
 *           contain the matrix  C,  except when  beta  is zero, in which
 *           case C need not be set on entry.
 *           On exit, the array  C  is overwritten by the  m by n  matrix
 *           ( alpha*op( A )*op( B ) + beta*C ).
 *
 *  LDC    - INTEGER.
 *           On entry, LDC specifies the first dimension of C as declared
 *           in  the  calling  (sub)  program.   LDC  must  be  at  least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *  Further Details
 *  ===============
 *
 *  Level 3 Blas routine.
 *
 *  -- Written on 8-February-1989.
 *     Jack Dongarra, Argonne National Laboratory.
 *     Iain Duff, AERE Harwell.
 *     Jeremy Du Croz, Numerical Algorithms Group Ltd.
 *     Sven Hammarling, Numerical Algorithms Group Ltd.
 *
 *  =====================================================================
 */
void sgemm_(character *TRANSA,character *TRANSB,integer *M,integer *N,integer *K,real *ALPHA,real *A,integer *LDA,real *B,integer *LDB,real *BETA,real *C, integer *LDC);
void dgemm_(character *TRANSA,character *TRANSB,integer *M,integer *N,integer *K,doublereal *ALPHA,doublereal *A,integer *LDA,doublereal *B, integer *LDB, doublereal *BETA,doublereal *C, integer *LDC);

/**
 *       SUBROUTINE DGEMV(TRANS,M,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
 *     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER INCX,INCY,LDA,M,N
      CHARACTER TRANS
 *     ..
 *     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
 *     ..
 *
 *  Purpose
 *  =======
 *
 *  DGEMV  performs one of the matrix-vector operations
 *
 *     y := alpha*A*x + beta*y,   or   y := alpha*A**T*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are vectors and A is an
 *  m by n matrix.
 *
 *  Arguments
 *  ==========
 *
 *  TRANS  - CHARACTER*1.
 *           On entry, TRANS specifies the operation to be performed as
 *           follows:
 *
 *              TRANS = 'N' or 'n'   y := alpha*A*x + beta*y.
 *
 *              TRANS = 'T' or 't'   y := alpha*A**T*x + beta*y.
 *
 *              TRANS = 'C' or 'c'   y := alpha*A**T*x + beta*y.
 *
 *           Unchanged on exit.
 *
 *  M      - INTEGER.
 *           On entry, M specifies the number of rows of the matrix A.
 *           M must be at least zero.
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the number of columns of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry, the leading m by n part of the array A must
 *           contain the matrix of coefficients.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, m ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( m - 1 )*abs( INCX ) ) otherwise.
 *           Before entry, the incremented array X must contain the
 *           vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of DIMENSION at least
 *           ( 1 + ( m - 1 )*abs( INCY ) ) when TRANS = 'N' or 'n'
 *           and at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ) otherwise.
 *           Before entry with BETA non-zero, the incremented array Y
 *           must contain the vector y. On exit, Y is overwritten by the
 *           updated vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  Further Details
 *  ===============
 *
 *  Level 2 Blas routine.
 *  The vector and matrix arguments are not referenced when N = 0, or M = 0
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 *
 *  =====================================================================
 */
void dgemv_(character *TRANS,integer *M,integer *N,doublereal *ALPHA, doublereal *A, integer *LDA, doublereal *X, integer *INCX, doublereal *BETA, doublereal *Y, integer *INCY);
void sgemv_(character *TRANS,integer *M,integer *N,real *ALPHA, real *A, integer *LDA, real *X, integer *INCX, real *BETA, real *Y, integer *INCY);

/**
 *       SUBROUTINE DSYMV(UPLO,N,ALPHA,A,LDA,X,INCX,BETA,Y,INCY)
 *     .. Scalar Arguments ..
      DOUBLE PRECISION ALPHA,BETA
      INTEGER INCX,INCY,LDA,N
      CHARACTER UPLO
 *     ..
 *     .. Array Arguments ..
      DOUBLE PRECISION A(LDA,*),X(*),Y(*)
 *     ..
 *
 *  Purpose
 *  =======
 *
 *  DSYMV  performs the matrix-vector  operation
 *
 *     y := alpha*A*x + beta*y,
 *
 *  where alpha and beta are scalars, x and y are n element vectors and
 *  A is an n by n symmetric matrix.
 *
 *  Arguments
 *  ==========
 *
 *  UPLO   - CHARACTER*1.
 *           On entry, UPLO specifies whether the upper or lower
 *           triangular part of the array A is to be referenced as
 *           follows:
 *
 *              UPLO = 'U' or 'u'   Only the upper triangular part of A
 *                                  is to be referenced.
 *
 *              UPLO = 'L' or 'l'   Only the lower triangular part of A
 *                                  is to be referenced.
 *
 *           Unchanged on exit.
 *
 *  N      - INTEGER.
 *           On entry, N specifies the order of the matrix A.
 *           N must be at least zero.
 *           Unchanged on exit.
 *
 *  ALPHA  - DOUBLE PRECISION.
 *           On entry, ALPHA specifies the scalar alpha.
 *           Unchanged on exit.
 *
 *  A      - DOUBLE PRECISION array of DIMENSION ( LDA, n ).
 *           Before entry with  UPLO = 'U' or 'u', the leading n by n
 *           upper triangular part of the array A must contain the upper
 *           triangular part of the symmetric matrix and the strictly
 *           lower triangular part of A is not referenced.
 *           Before entry with UPLO = 'L' or 'l', the leading n by n
 *           lower triangular part of the array A must contain the lower
 *           triangular part of the symmetric matrix and the strictly
 *           upper triangular part of A is not referenced.
 *           Unchanged on exit.
 *
 *  LDA    - INTEGER.
 *           On entry, LDA specifies the first dimension of A as declared
 *           in the calling (sub) program. LDA must be at least
 *           max( 1, n ).
 *           Unchanged on exit.
 *
 *  X      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCX ) ).
 *           Before entry, the incremented array X must contain the n
 *           element vector x.
 *           Unchanged on exit.
 *
 *  INCX   - INTEGER.
 *           On entry, INCX specifies the increment for the elements of
 *           X. INCX must not be zero.
 *           Unchanged on exit.
 *
 *  BETA   - DOUBLE PRECISION.
 *           On entry, BETA specifies the scalar beta. When BETA is
 *           supplied as zero then Y need not be set on input.
 *           Unchanged on exit.
 *
 *  Y      - DOUBLE PRECISION array of dimension at least
 *           ( 1 + ( n - 1 )*abs( INCY ) ).
 *           Before entry, the incremented array Y must contain the n
 *           element vector y. On exit, Y is overwritten by the updated
 *           vector y.
 *
 *  INCY   - INTEGER.
 *           On entry, INCY specifies the increment for the elements of
 *           Y. INCY must not be zero.
 *           Unchanged on exit.
 *
 *  Further Details
 *  ===============
 *
 *  Level 2 Blas routine.
 *  The vector and matrix arguments are not referenced when N = 0, or M = 0
 *
 *  -- Written on 22-October-1986.
 *     Jack Dongarra, Argonne National Lab.
 *     Jeremy Du Croz, Nag Central Office.
 *     Sven Hammarling, Nag Central Office.
 *     Richard Hanson, Sandia National Labs.
 */
void dsymv_(character *UPLO, integer *N, doublereal *ALPHA, doublereal *A, integer *LDA, doublereal *X, integer *INCX, doublereal *BETA, doublereal *Y, integer *INCY);
void ssymv_(character *UPLO, integer *N, real *ALPHA, real *A, integer *LDA, real *X, integer *INCX, real *BETA, real *Y, integer *INCY);

} // end of extern C

} // anonymous namespace
#endif
#endif

#ifdef HAVE_LIBBLAS
namespace approx{

/**
 * VECTOR INNER PRODUCTS
 */

//float dot(const std::vector<float> &x, const std::vector<float> &y){
//	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
//	integer N=x.size();
//	integer INC=1;
//
//	// const-casting because BLAS won't change the vectors anyways, C++ <> FORTRAN
//	std::vector<float> &xnoconst=const_cast<std::vector<float>& >(x);
//	std::vector<float> &ynoconst=const_cast<std::vector<float>& >(y);
//
//	float result=sdot_(&N,&xnoconst[0],&INC,&ynoconst[0],&INC);
//	return result;
//}
//
//double dot(const std::vector<double> &x, const std::vector<double> &y){
//	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
//	integer N=x.size();
//	integer INC=1;
//
//	// const-casting because BLAS won't change the vectors anyways, C++ <> FORTRAN
//	std::vector<double> &xnoconst=const_cast<std::vector<double>& >(x);
//	std::vector<double> &ynoconst=const_cast<std::vector<double>& >(y);
//
//	float result=ddot_(&N,&xnoconst[0],&INC,&ynoconst[0],&INC);
//	return result;
//}

/**
 * GENERAL MATRIX-MATRIX PRODUCT
 */

void gemm(const vector<float> &A, bool atrans, const vector<float> &B, bool btrans, size_t arows, vector<float> &C, float alpha){
	size_t acols=A.size()/arows, brows=acols, bcols=B.size()/brows;
	C.resize(arows*bcols);

	character TRANSA, TRANSB;
	integer M=arows, N=bcols, K=acols, LDA, LDB, LDC=M;
	if(atrans){
		TRANSA=*"T";
		LDA=K;
	}else{
		TRANSA=*"N";
		LDA=M;
	}
	if(btrans){
		TRANSB=*"T";
		LDB=N;
	}else{
		TRANSB=*"N";
		LDB=K;
	}
	real beta=0;

	std::vector<float> &Anoconst=const_cast<std::vector<float>& >(A);
	std::vector<float> &Bnoconst=const_cast<std::vector<float>& >(B);

	sgemm_(&TRANSA,&TRANSB,&M,&N,&K,&alpha,&Anoconst[0],&LDA,&Bnoconst[0],&LDB,&beta,&C[0],&LDC);
}

void  gemm(const vector<double> &A, bool atrans, const vector<double> &B, bool btrans, size_t arows, vector<double> &C, double alpha){
	size_t acols=A.size()/arows, brows=acols, bcols=B.size()/brows;
	C.resize(arows*bcols);

	character TRANSA, TRANSB;
	integer M=arows, N=bcols, K=acols, LDA, LDB, LDC=M;
	if(atrans){
		TRANSA=*"T";
		LDA=K;
	}else{
		TRANSA=*"N";
		LDA=M;
	}
	if(btrans){
		TRANSB=*"T";
		LDB=N;
	}else{
		TRANSB=*"N";
		LDB=K;
	}
	doublereal beta=0;

	std::vector<double> &Anoconst=const_cast<std::vector<double>& >(A);
	std::vector<double> &Bnoconst=const_cast<std::vector<double>& >(B);

	dgemm_(&TRANSA,&TRANSB,&M,&N,&K,&alpha,&Anoconst[0],&LDA,&Bnoconst[0],&LDB,&beta,&C[0],&LDC);

}

/**
 * GENERAL MATRIX-VECTOR PRODUCT
 */

void gemv(const vector<float> &A, bool trans, const vector<float> &v, float alpha, vector<float> &result){
	size_t cols=v.size(), rows=A.size()/cols;
	result.resize(rows);

	character TRANS;
	integer M, N, INCX=1, INCY=1, LDA;
	if(trans){
		TRANS=*"T";
		M=cols;
		N=rows;
	}else{
		TRANS=*"N";
		M=rows;
		N=cols;
	}
	LDA=M;

	float beta=1.0f;
	std::vector<float> &Anoconst=const_cast<std::vector<float>& >(A);
	std::vector<float> &vnoconst=const_cast<std::vector<float>& >(v);

	sgemv_(&TRANS,&M,&N,&alpha,&Anoconst[0],&LDA,&vnoconst[0],&INCX,&beta,&result[0],&INCY);
}

void gemv(const vector<double> &A, bool trans, const vector<double> &v, double alpha, vector<double> &result){
	size_t cols=v.size(), rows=A.size()/cols;
	result.resize(rows);

	character TRANS;
	integer M, N, INCX=1, INCY=1, LDA;
	if(trans){
		TRANS=*"T";
		M=cols;
		N=rows;
	}else{
		TRANS=*"N";
		M=rows;
		N=cols;
	}
	LDA=M;

	double beta=1;
	std::vector<double> &Anoconst=const_cast<std::vector<double>& >(A);
	std::vector<double> &vnoconst=const_cast<std::vector<double>& >(v);

	dgemv_(&TRANS,&M,&N,&alpha,&Anoconst[0],&LDA,&vnoconst[0],&INCX,&beta,&result[0],&INCY);

}

/**
 * SYMMETRIC MATRIX-VECTOR PRODUCT
 */

void symv(const vector<float> &M, const vector<float> &v, float alpha, vector<float> &result){
	result.resize(v.size(),0);

	character UPLO=*"U";
	integer N=v.size(), LDA=v.size(), INCX=1, INCY=1;
	real BETA=0;

	std::vector<float> &Mnoconst=const_cast<std::vector<float>& >(M);
	std::vector<float> &vnoconst=const_cast<std::vector<float>& >(v);

	ssymv_(&UPLO, &N, &alpha, &Mnoconst[0], &LDA, &vnoconst[0], &INCX, &BETA, &result[0], &INCY);

}

void symv(const vector<double> &M, const vector<double> &v, double alpha, vector<double> &result){
	result.resize(v.size(),0);

	character UPLO=*"U";
	integer N=v.size(), LDA=v.size(), INCX=1, INCY=1;
	doublereal BETA=0;

	std::vector<double> &Mnoconst=const_cast<std::vector<double>& >(M);
	std::vector<double> &vnoconst=const_cast<std::vector<double>& >(v);

	dsymv_(&UPLO, &N, &alpha, &Mnoconst[0], &LDA, &vnoconst[0], &INCX, &BETA, &result[0], &INCY);

}

} // approx namespace

#endif

#ifdef HAVE_LIBATLAS
namespace approx{

/**
 * VECTOR INNER PRODUCTS
 */

//float dot(const std::vector<float> &x, const std::vector<float> &y){
//	assert(x.size()==y.size() && "Attempting to compute inner product of vectors of unequal lengths!");
//	return cblas_sdot(x.size(),&x[0],1,&y[0],1);
//}

/**
 * GENERAL MATRIX-MATRIX PRODUCT
 */

void gemm(const vector<float> &A, bool atrans, const vector<float> &B, bool btrans, size_t arows, vector<float> &C, float alpha){
	size_t acols=A.size()/arows, brows=acols, bcols=B.size()/brows;
	C.resize(arows*bcols);

	enum CBLAS_TRANSPOSE transa, transb;

	int M=arows, N=bcols, K=acols, LDA, LDB, LDC=M;
	if(atrans){
		transa=CblasTrans;
		LDA=K;
	}else{
		transa=CblasNoTrans;
		LDA=M;
	}
	if(btrans){
		transb=CblasTrans;
		LDB=N;
	}else{
		transb=CblasNoTrans;
		LDB=K;
	}
	float beta=0;

	cblas_sgemm(CblasColMajor,transa,transb,arows,bcols,acols,alpha,&A[0],LDA,&B[0],LDB,beta,&C[0],LDC);
}

/**
 * GENERAL MATRIX-VECTOR PRODUCT
 */

void gemv(const vector<float> &A, bool trans, const vector<float> &v, float alpha, vector<float> &result){
	size_t cols=v.size(), rows=A.size()/cols;
	result.resize(rows);

	enum CBLAS_TRANSPOSE transp;
	int M, N, INCX=1, INCY=1, LDA;
	if(trans){
		transp=CblasTrans;
		M=cols;
		N=rows;
	}else{
		transp=CblasNoTrans;
		M=rows;
		N=cols;
	}
	LDA=M;

	float beta=1.0f;

	cblas_sgemv(CblasColMajor,transp,M,N,alpha,&A[0],LDA,&v[0],INCX,beta,&result[0],INCY);
}


/**
 * SYMMETRIC MATRIX-VECTOR PRODUCT
 */

void symv(const vector<float> &M, const vector<float> &v, float alpha, vector<float> &result){
	result.resize(v.size(),0);

	int N=v.size(), LDA=v.size(), INCX=1, INCY=1;
	float beta=0.0f;

	cblas_ssymv(CblasColMajor,CblasUpper,N,alpha,&M[0],LDA,&v[0],INCX,beta,&result[0],INCY);
}

} // approx namespace

#endif

