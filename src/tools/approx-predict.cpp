/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * approx-predict.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.h"
#include "Util.h"
#include "Approximation.h"
#include "config.h"
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <errno.h>
#include <vector>

using std::vector;
using std::auto_ptr;
using std::string;
using namespace approx;

std::string toolname("approx-predict");

/**
 * Reads a data instance from is. is should contain the remainder of the input line.
 */
inline void read_instance_sparse(std::istream &is, vector<float> &instance, size_t d){
	instance.clear();
	instance.resize(d,0);

	string chunk;
	size_t index;

	while(is.good()){

		chunk.clear();
		getline(is,chunk,*":");
		index=strtoul(chunk.c_str(),NULL,0); // much faster than stringstream
		if(index > d) break;

		getline(is,chunk,*" ");
		instance[index-1]=strtof(chunk.c_str(),NULL);
	}
}

/**
 * Reads a data instance from is. is should contain the remainder of the input line.
 */
inline void read_instance_dense(std::istream &is, vector<float> &instance, size_t d, char &delim){
	instance.clear();
	instance.reserve(d);

	size_t i=0;
	string chunk;

	while(is.good()){
		getline(is,chunk,delim);
		std::back_inserter(instance)=strtof(chunk.c_str(),NULL);
		if(++i > d)
			break;
		chunk.clear();
	}
}

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Performs predictions for test instances in given data file, using an approximated model.\n"
			"Each output line contains the predicted label and decision value (classification)\n"
			"or the predicted function value (regression)."
			"\n\n"
			"Options:\n"
	), helpfooter("");

	// intialize arguments
	std::deque<CLI::BaseArgument*> allargs;
	string description, keyword;
	vector<string> multilinedesc;

	keyword="--help";
	CLI::SilentFlagArgument help(keyword);
	allargs.push_back(&help);

	keyword="--h";
	CLI::SilentFlagArgument help2(keyword);
	allargs.push_back(&help2);

	keyword="--version";
	CLI::SilentFlagArgument version(keyword);
	allargs.push_back(&version);

	keyword="--v";
	CLI::SilentFlagArgument version2(keyword);
	allargs.push_back(&version2);

	keyword="-data";
	description="test data file";
	CLI::Argument<string> datafname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&datafname);

	keyword="-model";
	description="model file";
	CLI::Argument<string> modelfname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&modelfname);

	keyword="-o";
	description="output file";
	CLI::Argument<string> ofname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofname);

	description = "data file in dense format";
	keyword = "-dense";
	CLI::FlagArgument dense(description,keyword,false);
	allargs.push_back(&dense);

	keyword="-delim";
	description="delimiter for dense format (default: ',')";
	CLI::Argument<char> delim(description,keyword,CLI::Argument<char>::Content(1,*","));
	allargs.push_back(&delim);

	description = "data file contains ground truth for performance assessment (default: off)";
	multilinedesc.push_back(description);
	description = "ground truth = labels (classification) or function values (regression)";
	multilinedesc.push_back(description);
	keyword = "-truth";
	CLI::FlagArgument labeled(multilinedesc,keyword,false);
	allargs.push_back(&labeled);

	description = "suppress warnings when surpassing approximation bounds";
	keyword = "-q";
	CLI::FlagArgument warn(description,keyword,true);
	allargs.push_back(&warn);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(!(datafname && ofname && modelfname))
		exit_with_help(allargs,helpheader,helpfooter,false);

	std::ifstream modelfile(modelfname[0].c_str());
	auto_ptr<approx::Model> approx=approx::Model::load(modelfile, warn.value());
	modelfile.close();

	if(approx.get()==NULL){
		std::cerr << "Unable to load approximated model!" << std::endl;
		return EXIT_FAILURE;
	}

	std::ifstream ifile(datafname[0].c_str());

	char delimiter=*" ";
	if(dense.value())
		delimiter=delim[0];

	size_t d=approx->dimensionality();

	string line;
	std::istringstream iss;

	double performance=0;
	vector<float> instance;
	size_t numinstances=0;

	std::ofstream outfile;
	outfile.open(ofname[0].c_str());
	if(const approx::Classifier *classifier=dynamic_cast<const approx::Classifier*>(approx.get())){
		// doing classification
		std::pair<Classifier::Label,float> prediction;
		size_t numcorrect=0;
		if(labeled.value()){
			// data is labeled
			approx::Classifier::Label label;
			if(dense.value()){
				// data is dense
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;


					// fixme: work with int labels instead of strings
//					label.clear();
					iss.clear();
					iss.str(line);

					// read label
//					getline(iss,label,delimiter);
					iss >> label;

					// read instance
					read_instance_dense(iss,instance,d,delimiter);

					// predict
					prediction=classifier->classify(instance);

					// write output
					outfile << prediction.first << " " << prediction.second << std::endl;
					if(prediction.first == label)
						numcorrect++;
					numinstances++;
				}
			}else{
				// data is sparse
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

//					label.clear();
					iss.clear();
					iss.str(line);

					// read label
//					getline(iss,label,delimiter);
					iss >> label;

					// read instance
					read_instance_sparse(iss,instance,d);

					// predict
					prediction=classifier->classify(instance);

					// write output
					outfile << prediction.first << " " << prediction.second << std::endl;
					if(prediction.first == label) numcorrect++;
					numinstances++;
				}
			}
			performance=100.0f*numcorrect/numinstances;
			std::cout << "Accuracy = " << performance << "% (" << numcorrect << "/" << numinstances;
			std::cout << ") (classification)" << std::endl;
		}else{
			// data is unlabeled
			if(dense.value()){
				// data is dense
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);

					// read instance
					read_instance_dense(iss,instance,d,delimiter);

					// predict
					prediction=classifier->classify(instance);

					// write output
					outfile << prediction.first << " " << prediction.second << std::endl;
				}
			}else{
				// data is sparse
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);

					// read instance
					read_instance_sparse(iss,instance,d);

					// predict
					prediction=classifier->classify(instance);

					// write output
					outfile << prediction.first << " " << prediction.second << std::endl;
				}
			}
		}
	}else{
		// doing regression
		float prediction;
		if(labeled.value()){
			// data is labeled
			float label;
			string chunk;
			if(dense.value()){
				// data is dense
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);

					// read label
					chunk.clear();
					getline(iss,chunk,delimiter);
					label=strtof(chunk.c_str(),NULL);

					// read instance
					read_instance_dense(iss,instance,d,delimiter);

					// predict
					prediction=approx->evaluate(instance);

					// write output
					outfile << prediction << std::endl;

					// measure performance
					performance+=pow(label-prediction,2);
					numinstances++;
				}
			}else{
				// data is sparse
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);

					// read label
					chunk.clear();
					getline(iss,chunk,delimiter);
					label=strtof(chunk.c_str(),NULL);

					// read instance
					read_instance_sparse(iss,instance,d);

					// predict
					prediction=approx->evaluate(instance);

					// write output
					outfile << prediction << std::endl;

					// measure performance
					performance+=pow(label-prediction,2);
					numinstances++;
				}
			}
			performance=performance/numinstances; // MSE

			std::cout << "Mean squared error = " << performance << " (regression)" << std::endl;
			// todo output squared correlation coeff
		}else{
			// data is unlabeled
			if(dense.value()){
				// data is dense
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);
					// read instance
					read_instance_dense(iss,instance,d,delimiter);

					// predict
					prediction=approx->evaluate(instance);

					// write output
					outfile << prediction << std::endl;
				}
			}else{
				// data is sparse
				while(ifile.good()){
					line.clear();
					getline(ifile,line);
					if(line.size()==0) break;

					iss.clear();
					iss.str(line);

					// read instance
					read_instance_sparse(iss,instance,d);

					// predict
					prediction=approx->evaluate(instance);

					// write output
					outfile << prediction << std::endl;
				}
			}
		}
	}

	outfile.close();
	ifile.close();
	return 0;
}
