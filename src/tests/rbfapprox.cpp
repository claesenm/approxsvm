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
 * rbfapprox.cpp
 *
 *      Author: Marc Claesen
 */

#define DEBUGMATRIXMATH

#include "config.h"
#include "Models.h"
#include "Kernel.h"
#include "SparseVector.h"
#include "Approximation.h"
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <stdlib.h>
#include <memory>

using std::auto_ptr;
using std::string;
using std::vector;
using namespace ensemble;

template <typename T>
string typeName(T t){ return string("unknown"); }
string typeName(int t){ return string("int"); }
string typeName(double t){ return string("double"); }
string typeName(float t){ return string("float"); }

int main(int argc, char **argv)
{
	auto_ptr<std::deque<std::pair<int,double> > > svcontent(new std::deque<std::pair<int,double> >());
	svcontent->push_back(std::make_pair(1,0.1));
	svcontent->push_back(std::make_pair(2,0.2));
	svcontent->push_back(std::make_pair(3,0.3));
	SparseVector *v1=new SparseVector(svcontent); // [1 2 3]
	svcontent.reset(new std::deque<std::pair<int,double> >());
	svcontent->push_back(std::make_pair(1,0.1));
	svcontent->push_back(std::make_pair(2,0.2));
	svcontent->push_back(std::make_pair(4,0.4)); // [1 2 0 4]
	SparseVector *v2=new SparseVector(svcontent); // [1 2 3]

	auto_ptr<SVMModel::SV_container> SVs(new SVMModel::SV_container(2));
	SVs->at(0)=v1;
	SVs->at(1)=v2;

	auto_ptr<SVMModel::Weights> weights(new SVMModel::Weights(2,1));
	weights->at(1)=-1;

	auto_ptr<SVMModel::Classes> classes(new SVMModel::Classes());
	classes->resize(2);
	classes->at(0)=std::make_pair(std::string("pos"),1);
	classes->at(1)=std::make_pair(std::string("neg"),1);

	std::vector<double> constants(1,1);

	auto_ptr<Kernel> kernel(new RBFKernel(1));
	auto_ptr<SVMModel> model(new SVMModel(SVs, weights, classes, constants, kernel));

	std::cout << *model << std::endl;

	bool globalerr=false;

	auto_ptr<approx::ClassifierRBF<float> > approx(new approx::ClassifierRBF<float>(*model));

	std::vector<float> z(4,0);
	z[1]=0.1;
	z[2]=-0.1;
	z[3]=0.5;

	std::pair<string,float> pred=approx->predict(z);
	std::cout << "prediction: " << pred.first << " " << pred.second << std::endl;

	if(globalerr) exit(EXIT_FAILURE);
	else exit(EXIT_SUCCESS);
}
