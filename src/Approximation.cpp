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
 * Approximation.cpp
 *
 *      Author: Marc Claesen
 */

#include "Approximation.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <math.h>
#include <functional>
#include <numeric>
#include <iostream>

using std::vector;
using std::pair;
using std::auto_ptr;
using std::string;

namespace approx{

std::ostream &Classifier::write(std::ostream &os, bool header) const{
	os << "approximated SVM classifier" << std::endl;
	os << "labels " << positive << " " << negative << std::endl;
	os << "threshold " << threshold << std::endl;
	regressor->write(os,false);
	return os;
}


Classifier::Classifier(const svm_model &model):Model(),threshold(0.0f){
	if(model.param.svm_type > 2) // no classifier
		assert(false && "Attempting to construct classifier from non-classifier LIBSVM model!");
	assert(model.nr_class==2 && "Unable to approximate multiclass modes!");

	positive=model.label[0];
	negative=model.label[1];

	assert(model.param.kernel_type==2 && "Unable to approximate non-RBF model!");
	regressor.reset(new RBFModel(model));
	type=regressor->getType();
	d=regressor->dimensionality();
}

RBFModel::RBFModel(const svm_model &model):Model(),warn(true){
	if(model.param.svm_type >= 0 && model.param.svm_type < 3) // classifier
		assert(model.nr_class==2 && "Unable to approximate multiclass models!");

	if(model.param.svm_type==0) type=SVM_TYPE::CSVC;
	if(model.param.svm_type==1) type=SVM_TYPE::NUSVM;
	if(model.param.svm_type==2) type=SVM_TYPE::OSVM;
	if(model.param.svm_type==3) type=SVM_TYPE::EPSSVR;
	if(model.param.svm_type==4) type=SVM_TYPE::NUSVR;

	/**
	 * model-specific interface
	 */

	b=-model.rho[0];
	gamma=model.param.gamma;

	vector<float> X, coefs;
	computeIntermediates(model,X,coefs);

	/**
	 *  model-agnostic computations
	 */

	computeSolution(X,coefs);
}

void RBFModel::computeSolution(const vector<float> &X, const vector<float> &coefs){
	// set parameter c
	c=std::accumulate(coefs.begin(),coefs.end(),float(0));

	// compute and set v=X*w
	gemv(X,false,coefs,float(2*gamma),v);

	// compute and set M
	vector<float> XD(X);
	dmul(XD,false,coefs,float(2*gamma*gamma));
	gemm(XD,false,X,true,d,M);
}

void RBFModel::computeIntermediates(const svm_model &model, vector<float> &X, vector<float> &coefs){
	size_t nSV=model.l;
	coefs.resize(nSV,0);

	maxnorm=0;
	float norm=0;
	size_t maxd=0;

	for(size_t i=0;i<nSV;++i){
		norm=0;
		for(size_t j=0;model.SV[i][j].index > 0;++j){
			d=(size_t)model.SV[i][j].index;
			norm+=model.SV[i][j].value*model.SV[i][j].value;
		}

		if(d > maxd)
			maxd=d;

		coefs[i]=model.sv_coef[0][i]*exp(-gamma*norm);
		if(norm > maxnorm) maxnorm=norm;
	}
	d=maxd;

	X.resize(maxd*nSV,0);
	for(size_t i=0;i<nSV;++i){
		for(size_t j=0;model.SV[i][j].index > 0;++j){
			X[i*maxd+model.SV[i][j].index-1]=float(model.SV[i][j].value);
		}
	}
}

float RBFModel::evaluate(const vector<float> &z) const{

//	static const float cutoff=0.125;
	static const float cutoff=1.0f/16;

	float expval=dot(z,z); 			// ||z||^2
	if(warn && expval*maxnorm*gamma*gamma > cutoff)
		std::cerr << "Warning: prediction may be out of approximation bounds!" << std::endl;

	expval=exp(-gamma*expval); 	// exp(-gamma*||z||^2)
	const vector<float> *zref=&z;
	vector<float> zcut;

	if(z.size()>d){ // if z is of higher dimension than any SV, cut off the last dimensions
		std::copy(z.begin(),z.begin()+d,std::back_inserter(zcut));
		zref=&zcut;
	}
	float result=c+dot(v,*zref); 	// constant + first order term

//	vector<float> Mz;
//	symv(M,*zref,float(1),Mz);
//	result+=dot(Mz,*zref); 		// + second-order term
	result+=vtMv(M,*zref); 		// + second-order term (fallback)
	return expval*(result)+b; 	// final approximated decision value
}

std::ostream &RBFModel::write(std::ostream &os, bool header) const{
	Model::write(os, header);
	os << "kernel_type rbf" << std::endl;
	os << "gamma " << gamma << std::endl;
	os << "maxnorm " << maxnorm << std::endl;
	os << "b " << b << std::endl;
	os << "c " << c << std::endl;
	os << "v";
	for(std::vector<float>::const_iterator I=v.begin(),E=v.end();I!=E;++I)
		os << " " << *I;
	os << std::endl;
	os << "M" << std::endl;
	printFortranMatrix(M,d,false,os);
	return os;
}

auto_ptr<Model> Model::load(std::istream &is, bool warn){ // fixme: handle regression
	string line;
	getline(is,line);
	if(line.compare("approximated SVM classifier")==0){

		line.clear();
		is >> line;
		if(line.compare("labels")!=0) return auto_ptr<Model>(NULL);
		line.clear();

		Classifier::Label poslabel, neglabel;
		is >> poslabel;
		is >> neglabel;
		double threshold;

		getline(is,line); // remove trailing newline
		line.clear();
		is >> line;
		if(line.compare("threshold")!=0) return auto_ptr<Model>(NULL);
		is >> threshold;
		getline(is,line); // remove trailing newline

		line.clear();
		is >> line;
		if(line.compare("svm_type")!=0) return auto_ptr<Model>(NULL);

		int type=-1;
		line.clear();
		is >> line;
		if(line.compare("c_svc")==0) type=SVM_TYPE::CSVC;
		else if(line.compare("nu_svm")==0) type=SVM_TYPE::NUSVM;
		else if(line.compare("ls_svm")==0) type=SVM_TYPE::LSSVM;
		else if(line.compare("one_svm")==0) type=SVM_TYPE::OSVM;
		getline(is,line); // remove trailing newline

		line.clear();
		is >> line;
		assert(line.compare("d")==0 && "Invalid approximate model format!");
		size_t dim;
		is >> dim;
		getline(is,line); // trailing newline;

		line.clear();
		getline(is,line);
		if(line.compare("kernel_type rbf")!=0) return auto_ptr<Model>(NULL);

		auto_ptr<Model> regressor(new RBFModel(type,dim,is, warn));
		return auto_ptr<Model>(new Classifier(poslabel,neglabel,float(threshold),regressor));

	}else if(line.compare("approximated SVM regressor")==0){
		// read regressor
		line.clear();
		is >> line;
		if(line.compare("svm_type")!=0) return auto_ptr<Model>(NULL);

		int type=-1;
		line.clear();
		is >> line;
		if(line.compare("eps_svr")==0) type=SVM_TYPE::EPSSVR;
		else if(line.compare("nu_svr")==0) type=SVM_TYPE::NUSVR;
		else if(line.compare("ls_svm")==0) type=SVM_TYPE::LSSVM;
		getline(is,line); // remove trailing newline

		line.clear();
		is >> line;
		assert(line.compare("d")==0 && "Invalid approximate model format!");
		size_t dim;
		is >> dim;
		getline(is,line); // trailing newline;

		line.clear();
		getline(is,line);
		if(line.compare("kernel_type rbf")!=0) return auto_ptr<Model>(NULL);

		auto_ptr<Model> regressor(new RBFModel(type,dim,is,warn));
		return regressor;
	}
	return auto_ptr<Model>(NULL);
}

RBFModel::RBFModel(int type, size_t d, std::istream &is, bool warn)
:Model(type,d),warn(warn){

	string keyword;
	is >> keyword;
	assert(keyword.compare("gamma")==0 && "Invalid approximate RBF classifier format!");
	is >> gamma;

	keyword.clear();
	is >> keyword;
	assert(keyword.compare("maxnorm")==0 && "Invalid approximate RBF classifier format!");
	is >> maxnorm;

	keyword.clear();
	is >> keyword;
	assert(keyword.compare("b")==0 && "Invalid approximate RBF classifier format!");
	is >> b;

	keyword.clear();
	is >> keyword;
	assert(keyword.compare("c")==0 && "Invalid approximate RBF classifier format!");
	is >> c;

	keyword.clear();
	is >> keyword;
	assert(keyword.compare("v")==0 && "Invalid approximate RBF classifier format!");
	v.reserve(d);
	float val;
	for(size_t i=0;i<d;++i){
		is >> val;
		std::back_inserter(v)=val;
	}

	keyword.clear();
	is >> keyword;
	assert(keyword.compare("M")==0 && "Invalid approximate RBF classifier format!");
	M.reserve(d*d);

	for(size_t i=0;i<d*d;++i){
		is >> val;
		std::back_inserter(M)=val;
	}
}


auto_ptr<Model> Model::approximate(const svm_model &model){
	if(model.param.svm_type > 2) // no classifier
		if(model.param.kernel_type == 2){ // RBF regressor
			return auto_ptr<Model>(new RBFModel(model));
		}else
			return auto_ptr<Model>(NULL); // todo
	else{
		return auto_ptr<Model>(new Classifier(model));
	}
}

} // approx namespace
