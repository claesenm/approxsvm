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
 * test_matrixmath.cpp
 *
 *      Author: Marc Claesen
 */

#define DEBUGMATRIXMATH

#include "config.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include "MatrixMath.h"
#include <string>
#include <stdlib.h>

using std::string;
using std::vector;
using namespace approx;

template <typename T>
string typeName(T t){ return string("unknown"); }
string typeName(int t){ return string("int"); }
string typeName(double t){ return string("double"); }
string typeName(float t){ return string("float"); }

/**
 * Returns true if an error occurred during the test.
 */
template <typename T>
bool test_dot(){
	vector<T> a(2), b(2);
	a[0]=1;
	a[1]=2;
	b[0]=3;
	b[1]=4;

	T c=approx::dot(a,b), solution=11;
	if(c==solution)
		return false;

	T dummy=0;
	std::cerr << "Failed dot product for " << typeName(dummy) << "!" << std::endl;
	return true;
}

/**
 * Returns true if an error occurred during the test.
 */
template <typename T>
bool test_gemm(){
	bool err=false;
	T dummy=0;

	vector<T> M(8,0);
	M[1]=4;
	M[2]=1;
	M[3]=5;
	M[4]=2;
	M[5]=6;
	M[6]=3;
	M[7]=7;

	T alpha=1;
	size_t rows=2;
	std::vector<T> res;

	/**
	 * MATRIX * MATRIX
	 */

	// M1*M2
	approx::gemm(M,false,M,false,rows,res,alpha);
	const T M1M2[]={21,61,33,105};
	for(unsigned i=0;i<4;++i)
		if(M1M2[i]!=res[i]) err=true;
	if(err) std::cerr << "Failed matrix x matrix for " << typeName(dummy) << "!" << std::endl;

	/**
	 *  MATRIX * TRANSPOSE
	 */

	// M1*M1'
	approx::gemm(M,false,M,true,rows,res,alpha);
	const T M1M1t[]={14,38,38,126};
	for(unsigned i=0;i<4;++i)
		if(M1M1t[i]!=res[i]) err=true;
	if(err) std::cerr << "Failed matrix x transpose for " << typeName(dummy) << "!" << std::endl;

	/**
	 * TRANSPOSE * MATRIX
	 */

	rows=4;
	// M1'*M1
	approx::gemm(M,true,M,false,rows,res,alpha);

	// check solution
	const T M1tM1[]={16,20,24,28,20,26,32,38,24,32,40,48,28,38,48,58};
	for(unsigned i=0;i<16;++i)
		if(M1tM1[i]!=res[i]) err=true;
	if(err) std::cerr << "Failed transpose x matrix for " << typeName(dummy) << "!" << std::endl;

	/**
	 * TRANSPOSE * TRANSPOSE
	 */

	// M1t*M2t
	approx::gemm(M,true,M,true,rows,res,alpha);

	// check solution
	const T M1tM2t[]={8,10,12,14,24,34,44,54,12,16,20,24,28,40,52,64};
	for(unsigned i=0;i<16;++i)
		if(M1tM2t[i]!=res[i]) err=true;
	if(err) std::cerr << "Failed transpose x transpose for " << typeName(dummy) << "!" << std::endl;

	return err;
}

/**
 * Returns true if an error occurred during the test.
 */
template <typename T>
bool test_gemv(){
	bool err=false;
	T alpha=1, dummy=0;
	vector<T> M(6,0), v(2,1), result;
	M[1]=1;
	M[2]=2;		// M: 0 1 2 3 4 5
	M[3]=3;		// M=[0 2 4; 1 3 5]
	M[4]=4;		// Mt=[0 1; 2 3; 4 5]
	M[5]=5;
	v[1]=2;		// v2=[1 2]

	// test M'*v2
	approx::gemv(M,true,v,alpha,result);
	int solution2[]={2,8,14};
	for(unsigned i=0;i<3;++i){
		if(solution2[i]!=result[i]) err=true;
	}
	if(err) std::cerr << "Failed transpose x vector for " << typeName(dummy) << "!" << std::endl;

//	// test M*v1
//	std::cout << "M:" << std::endl;
//	printFortranMatrix(M,M.size()/v.size(),false);

	approx::gemv(M,false,v,alpha,result);
	int solution1[]={6,9,12};
	for(unsigned i=0;i<3;++i){
		if(solution1[i]!=result[i]) err=true;
//		std::cout << result[i] << " ";
	}
//	std::cout << std::endl;
	if(err) std::cerr << "Failed matrix x vector for " << typeName(dummy) << "!" << std::endl;

	return err;
}

int main(int argc, char **argv)
{
	bool globalerr=false;

	// DOT tests
	{
		globalerr = globalerr | test_dot<int>();
		globalerr = globalerr | test_dot<float>();
		globalerr = globalerr | test_dot<double>();
	}

	// GEMM tests
	{
		globalerr = globalerr | test_gemm<int>();
		globalerr = globalerr | test_gemm<float>();
		globalerr = globalerr | test_gemm<double>();
	}

	// GEMV tests
	{
		globalerr = globalerr | test_gemv<int>();
		globalerr = globalerr | test_gemv<float>();
		globalerr = globalerr | test_gemv<double>();
	}

	// SYMV tests
	{
		// todo
	}

	// DMUL test
	{
		// todo
	}

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
