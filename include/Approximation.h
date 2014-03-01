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

} // approx namespace

#endif /* APPROXIMATION_H_ */
