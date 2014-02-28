/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * approx-svm.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.h"
#include "Util.h"
#include "Approximation.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <errno.h>

using std::vector;
using std::deque;
using std::auto_ptr;
using std::string;
using namespace approx;

string toolname("approx-svm");

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Approximates the specified SVM model (LIBSVM).\n"
			"\n"
			"Options:\n"
	), helpfooter("");

	// intialize arguments
	std::deque<CLI::BaseArgument*> allargs;
	string description, keyword;

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

	keyword="-model";
	description="model which must be approximated";
	CLI::Argument<string> model(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&model);

	keyword="-o";
	description="output file";
	CLI::Argument<string> ofname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&ofname);

	description = "enables verbose mode, which outputs various information to stdout";
	keyword = "-v";
	CLI::FlagArgument verbose(description,keyword,false);
	allargs.push_back(&verbose);

	if(argc==1)
		exit_with_help(allargs,helpheader,helpfooter,true);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	bool validargs=true;
	if(!model.configured()){
		std::cerr << "Model file not specified (see -model)." << std::endl;
		validargs=false;
	}
	if(!ofname.configured()){
		std::cerr << "Output file not specified (see -o)." << std::endl;
		validargs=false;
	}
	if(!validargs)
		exit_with_err("Invalid command line arguments provided.");

	// read LIBSVM model
	auto_ptr<svm_model> libsvmmodel(svm_load_model(model[0].c_str()));
	std::ofstream ofile(ofname[0].c_str());
	auto_ptr<approx::Model> approx=approx::Model::approximate(*libsvmmodel);

	if(approx.get()==NULL){
		std::cerr << "Unable to approximate specified model!" << std::endl;
		return EXIT_FAILURE;
	}

	ofile << *approx;
	ofile.close();

	return EXIT_SUCCESS;
}
