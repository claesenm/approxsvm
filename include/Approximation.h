/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * Approximation.h
 *
 *      Author: Marc Claesen
 */

#ifndef APPROXIMATION_H_
#define APPROXIMATION_H_

#include "Util.h"
#include "svm.h"
#include "MatrixMath.h"
#include <algorithm>
#include <vector>
#include <cassert>
#include <math.h>
#include <functional>
#include <numeric>
#include <iostream>
#include <memory>

using std::vector;
using std::pair;
using std::auto_ptr;

namespace approx{

struct SVM_TYPE{
	enum {
		UNKNOWN=-1,		// should never occur
		CSVC=0,			// classic SVM classifier
		NUSVM=1,		// nu-SVM
		OSVM=2,			// one-class SVM
		EPSSVR=3,		// epsilon-SVR
		NUSVR=4,		// nu-SVR
		LSSVM=5		// least-squares SVM
	};
};

struct KERNEL_TYPE{
	enum {
		UNKNOWN=-1,
		RBF=0,
		POLY=1
	};
};

class Classifier;

class Model{
protected:
	int type;
	size_t d;

	virtual std::ostream &write(std::ostream &os, bool header=true) const{
		if(header) os << "approximated SVM regressor" << std::endl;

		os << "svm_type ";
		switch(type){
		case SVM_TYPE::CSVC: os << "c_svc";
		break;
		case SVM_TYPE::NUSVM: os << "nu_svm";
		break;
		case SVM_TYPE::OSVM: os << "one_svm";
		break;
		case SVM_TYPE::EPSSVR: os << "eps_svr";
		break;
		case SVM_TYPE::NUSVR: os << "nu_svr";
		break;
		case SVM_TYPE::LSSVM: os << "ls_svm";
		break;
		default: os << "unknown"; // should never happen
		break;
		}
		os << std::endl << "d " << d << std::endl;

		return os;
	};
	Model():d(0){};
	Model(int type):type(type),d(0){};
	Model(int type, size_t d):type(type),d(d){};

public:
	virtual bool isClassifier() const{ return false; }
	virtual ~Model(){};

	int getType() const{ return type; }
	size_t dimensionality() const{ return d; }

	/**
	 * Writes the approximated Model to os.
	 */
	friend std::ostream &operator<<(std::ostream &os, const Model &approx){
		return approx.write(os);
	}

	/**
	 * Reads an approximated Model from is. This is a factory method.
	 * If warn=false, no warnings are issued when approximation boundaries are not satisfied.
	 */
	static auto_ptr<Model> load(std::istream &is, bool warn=true);
	virtual float evaluate(const std::vector<float> &z) const=0;

	/**
	 * Attempts to approximate given model. This is a factory method.
	 */
	static auto_ptr<Model> approximate(const svm_model &model);

	friend class Classifier;
};


class Classifier : public Model{
public:
	typedef int Label;
protected:
	Label positive, negative;
	float threshold;
	auto_ptr<Model> regressor;

	virtual std::ostream &write(std::ostream &os, bool header=true) const;

public:
	Classifier(Label &positive, Label &negative, float threshold, auto_ptr<Model> regressor)
	:Model(regressor->getType(),regressor->dimensionality()),positive(positive),negative(negative),threshold(threshold),regressor(regressor){};
	Classifier(const svm_model &model);

	virtual inline float evaluate(const std::vector<float> &z) const{ return regressor->evaluate(z); };
	virtual std::pair<Label,float> classify(const std::vector<float> &z) const{
		float decvalue=evaluate(z);
		if(decvalue >= threshold) return std::make_pair(positive,decvalue);
		return std::make_pair(negative,decvalue);
	}
	virtual bool isClassifier() const{ return true; }
	virtual ~Classifier(){ regressor.reset(NULL); };
};

//std::ostream &Classifier::write(std::ostream &os, bool header) const{
//	os << "approximated SVM classifier" << std::endl;
//	os << "labels " << positive << " " << negative << std::endl;
//	os << "threshold " << threshold << std::endl;
//	regressor->write(os,false);
//	return os;
//}

class RBFModel:public Model{

private:
	// RBF parameter
	float gamma;

	// bias in SVM decision function
	float b;

	// constant coefficient of the first-order approximation
	float c;

	// coefficients of the first-order term in the approximation v^T*z
	vector<float> v;

	// coefficients of the second-order term in the approximation: z^T*M*z
	// stored in Fortran-style
	vector<float> M;

	// the maximum norm of all SVs used to construct this model.
	float maxnorm;

	// set to true if warnings must be output
	bool warn;

protected:

	/**
	 * Returns the dense support vector matrix in X and coefficients in coefs, based on given model.
	 * This will also set the internal parameter maxnorm.
	 *
	 * coefs[i]=alphay[i]*exp(-gamma*sqnorm(X[i]))
	 */
	void computeIntermediates(const svm_model &model, vector<float> &X, vector<float> &coefs); // interface to LIBSVM models

	/**
	 * Computes c, v and M based on intermediate results.
	 */
	void computeSolution(const vector<float> &X, const vector<float> &coefs);
	virtual std::ostream &write(std::ostream &os, bool header=true) const;

	//	RBFModel(int type, const ClassLabel &poslabel, const ClassLabel &neglabel, float threshold, std::istream &is);
	RBFModel(int type, size_t d, std::istream  &is, bool warn=true);

public:
	RBFModel(const svm_model &model);

	virtual float evaluate(const std::vector<float> &z) const;

	virtual ~RBFModel(){};

	friend class Model;
	friend class Classifier;
};

//Classifier::Classifier(const svm_model &model):Model(),threshold(0){
//	if(model.param.svm_type > 2) // no classifier
//		assert(false && "Attempting to construct classifier from non-classifier LIBSVM model!");
//	assert(model.nr_class==2 && "Unable to approximate multiclass modes!");
//
//	positive=model.label[0];
//	negative=model.label[1];
//
//	assert(model.param.kernel_type==2 && "Unable to approximate non-RBF model!");
//	regressor.reset(new RBFModel(model));
//	type=regressor->getType();
//	d=regressor->dimensionality();
//}
//
//RBFModel::RBFModel(const svm_model &model):Model(){
//	if(model.param.svm_type >= 0 && model.param.svm_type < 3) // classifier
//		assert(model.nr_class==2 && "Unable to approximate multiclass models!");
//
//	if(model.param.svm_type==0) type=SVM_TYPE::CSVC;
//	if(model.param.svm_type==1) type=SVM_TYPE::NUSVM;
//	if(model.param.svm_type==2) type=SVM_TYPE::OSVM;
//	if(model.param.svm_type==3) type=SVM_TYPE::EPSSVR;
//	if(model.param.svm_type==4) type=SVM_TYPE::NUSVR;
//
//	/**
//	 * model-specific interface
//	 */
//
//	b=-model.rho[0];
//	gamma=model.param.gamma;
//
//	vector<float> X, coefs;
//	computeIntermediates(model,X,coefs);
//
//	/**
//	 *  model-agnostic computations
//	 */
//
//	computeSolution(X,coefs);
//}
//
//void RBFModel::computeSolution(const vector<float> &X, const vector<float> &coefs){
//	// set parameter c
//	c=std::accumulate(coefs.begin(),coefs.end(),float(0));
//
//	// compute and set v=X*w
//	gemv(X,false,coefs,float(2*gamma),v);
//
//	// compute and set M
//	vector<float> XD(X);
//	dmul(XD,false,coefs,float(2*gamma*gamma));
//	gemm(XD,false,X,true,d,M);
//}
//
//
//void RBFModel::computeIntermediates(const ensemble::SVMModel &model, vector<float> &X, vector<float> &coefs){
//	size_t nSV=model.size();
//	coefs.resize(nSV,0);
//
//	maxnorm=0;
//	float norm=0;
//	size_t maxd=0;
//	ensemble::SVMModel::const_iterator Isv=model.begin(),Esv=model.end();
//	ensemble::SVMModel::const_weight_iter Iw=model.weight_begin(), Ew=model.weight_end();
//	for(size_t i=0;i<nSV;++i,++Isv,++Iw){
//		d=(*Isv)->rbegin()->first;
//		if(d > maxd)
//			maxd=d;
//		norm=squaredNorm(**Isv);
//		coefs[i]=(*Iw) * exp(-gamma*norm);
//		if(norm > maxnorm) maxnorm=norm;
//	}
//
//	d=maxd;
//
//	X.resize(maxd*nSV,0);
//	size_t colnum=0;
//	for(Isv=model.begin();Isv!=Esv;++Isv,++colnum){
//		for(ensemble::SparseVector::const_iterator Iv=(*Isv)->begin(),Ev=(*Isv)->end();Iv!=Ev;++Iv){
//			X[colnum*maxd+Iv->first-1]=float(Iv->second); // SparseVector data always start at index 1
//		}
//	}
//}
//
//void RBFModel::computeIntermediates(const svm_model &model, vector<float> &X, vector<float> &coefs){
//	size_t nSV=model.l;
//	coefs.resize(nSV,0);
//
//	maxnorm=0;
//	float norm=0;
//	size_t maxd=0;
//
//	for(size_t i=0;i<nSV;++i){
//		norm=0;
//		for(size_t j=0;model.SV[i][j].index > 0;++j){
//			d=(size_t)model.SV[i][j].index;
//			norm+=model.SV[i][j].value*model.SV[i][j].value;
//		}
//
//		if(d > maxd)
//			maxd=d;
//
//		coefs[i]=model.sv_coef[0][i]*exp(-gamma*norm);
//		if(norm > maxnorm) maxnorm=norm;
//	}
//	d=maxd;
//
//	X.resize(maxd*nSV,0);
//	for(size_t i=0;i<nSV;++i){
//		for(size_t j=0;model.SV[i][j].index > 0;++j){
//			X[i*maxd+model.SV[i][j].index-1]=float(model.SV[i][j].value);
//		}
//	}
//}
//
//float RBFModel::evaluate(const vector<float> &z) const{
//
//	static const float cutoff=0.125;
//
//	float expval=dot(z,z); 			// ||z||^2
//	if(expval*maxnorm*gamma*gamma > cutoff)
//		std::cerr << "Warning: prediction may be out of approximation bounds!" << std::endl;
//
//	expval=exp(-gamma*expval); 	// exp(-gamma*||z||^2)
//	const vector<float> *zref=&z;
//	vector<float> zcut;
//
//	if(z.size()>d){ // if z is of higher dimension than any SV, cut off the last dimensions
//		std::copy(z.begin(),z.begin()+d,std::back_inserter(zcut));
//		zref=&zcut;
//	}
//	float result=c+dot(v,*zref); 	// constant + first order term
//
//	vector<float> Mz;
//	symv(M,*zref,float(1),Mz);
//	result+=dot(Mz,*zref); 		// + second-order term
//	return expval*(result)+b; 	// final approximated decision value
//}
//
//std::ostream &RBFModel::write(std::ostream &os, bool header) const{
//	Model::write(os, header);
//	os << "kernel_type rbf" << std::endl;
//	os << "gamma " << gamma << std::endl;
//	os << "maxnorm " << maxnorm << std::endl;
//	os << "b " << b << std::endl;
//	os << "c " << c << std::endl;
//	os << "v";
//	for(std::vector<float>::const_iterator I=v.begin(),E=v.end();I!=E;++I)
//		os << " " << *I;
//	os << std::endl;
//	os << "M" << std::endl;
//	printFortranMatrix(M,d,false,os);
//	return os;
//}
//
//auto_ptr<Model> Model::load(std::istream &is){ // fixme: handle regression
//	string line;
//	getline(is,line);
//	if(line.compare("approximated SVM classifier")==0){
//
//		line.clear();
//		is >> line;
//		if(line.compare("labels")!=0) return auto_ptr<Model>(NULL);
//		line.clear();
//
//		Classifier::Label poslabel, neglabel;
//		is >> poslabel;
//		is >> neglabel;
//		double threshold;
//
//		getline(is,line); // remove trailing newline
//		line.clear();
//		is >> line;
//		if(line.compare("threshold")!=0) return auto_ptr<Model>(NULL);
//		is >> threshold;
//		getline(is,line); // remove trailing newline
//
//		line.clear();
//		is >> line;
//		if(line.compare("svm_type")!=0) return auto_ptr<Model>(NULL);
//
//		int type=-1;
//		line.clear();
//		is >> line;
//		if(line.compare("c_svc")==0) type=SVM_TYPE::CSVC;
//		else if(line.compare("nu_svm")==0) type=SVM_TYPE::NUSVM;
//		else if(line.compare("ls_svm")==0) type=SVM_TYPE::LSSVM;
//		else if(line.compare("one_svm")==0) type=SVM_TYPE::OSVM;
//		getline(is,line); // remove trailing newline
//
//		line.clear();
//		is >> line;
//		assert(line.compare("d")==0 && "Invalid approximate model format!");
//		size_t dim;
//		is >> dim;
//		getline(is,line); // trailing newline;
//
//		line.clear();
//		getline(is,line);
//		if(line.compare("kernel_type rbf")!=0) return auto_ptr<Model>(NULL);
//
//		auto_ptr<Model> regressor(new RBFModel(type,dim,is));
//		return auto_ptr<Model>(new Classifier(poslabel,neglabel,float(threshold),regressor));
//	}else if(line.compare("approximated SVM regressor")==0){
//		// read regressor
//		line.clear();
//		is >> line;
//		if(line.compare("svm_type")!=0) return auto_ptr<Model>(NULL);
//
//		int type=-1;
//		line.clear();
//		is >> line;
//		if(line.compare("eps_svr")==0) type=SVM_TYPE::EPSSVR;
//		else if(line.compare("nu_svr")==0) type=SVM_TYPE::NUSVR;
//		else if(line.compare("ls_svm")==0) type=SVM_TYPE::LSSVM;
//		getline(is,line); // remove trailing newline
//
//		line.clear();
//		is >> line;
//		assert(line.compare("d")==0 && "Invalid approximate model format!");
//		size_t dim;
//		is >> dim;
//		getline(is,line); // trailing newline;
//
//		line.clear();
//		getline(is,line);
//		if(line.compare("kernel_type rbf")!=0) return auto_ptr<Model>(NULL);
//
//		auto_ptr<Model> regressor(new RBFModel(type,dim,is));
//		return regressor;
//	}
//	return auto_ptr<Model>(NULL);
//}
//
//RBFModel::RBFModel(int type, size_t d, std::istream &is)
//:Model(type,d){
//
//	string keyword;
//	is >> keyword;
//	assert(keyword.compare("gamma")==0 && "Invalid approximate RBF classifier format!");
//	is >> gamma;
//
//	keyword.clear();
//	is >> keyword;
//	assert(keyword.compare("maxnorm")==0 && "Invalid approximate RBF classifier format!");
//	is >> maxnorm;
//
//	keyword.clear();
//	is >> keyword;
//	assert(keyword.compare("b")==0 && "Invalid approximate RBF classifier format!");
//	is >> b;
//
//	keyword.clear();
//	is >> keyword;
//	assert(keyword.compare("c")==0 && "Invalid approximate RBF classifier format!");
//	is >> c;
//
//	keyword.clear();
//	is >> keyword;
//	assert(keyword.compare("v")==0 && "Invalid approximate RBF classifier format!");
//	v.reserve(d);
//	float val;
//	for(size_t i=0;i<d;++i){
//		is >> val;
//		std::back_inserter(v)=val;
//	}
//
//	keyword.clear();
//	is >> keyword;
//	assert(keyword.compare("M")==0 && "Invalid approximate RBF classifier format!");
//	M.reserve(d*d);
//
//	for(size_t i=0;i<d*d;++i){
//		is >> val;
//		std::back_inserter(M)=val;
//	}
//}
//
//
//auto_ptr<Model> Model::approximate(const svm_model &model){
//	if(model.param.svm_type > 2) // no classifier
//		if(model.param.kernel_type == 2){ // RBF regressor
//			return auto_ptr<Model>(new RBFModel(model));
//		}else
//			return auto_ptr<Model>(NULL); // todo
//	else{
//		return auto_ptr<Model>(new Classifier(model));
//	}
//}

} // approx namespace

#endif /* APPROXIMATION_H_ */
