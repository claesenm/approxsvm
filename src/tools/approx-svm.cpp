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
			"Approximates the specified SVM model (LIBSVM/LS-SVM).\n"
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
