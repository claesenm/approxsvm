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
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <errno.h>

using std::string;
using namespace approx;

std::string toolname("approx-analyse");

/**
 * Reads a data instance from is. is should contain the remainder of the input line.
 */
inline std::pair<size_t, float> read_instance_sparse(std::istream &is){
	std::pair<size_t,float> result(0,0.0f);

	string chunk;
	float val;

	while(is.good()){
		chunk.clear();
		getline(is,chunk,*":");
		if(chunk.empty()) break;
		result.first=strtoul(chunk.c_str(),NULL,0); // much faster than stringstream

		getline(is,chunk,*" ");
		val=strtof(chunk.c_str(),NULL);
		result.second+=val*val;
	}

	return result;
}

/**
 * Reads a data instance from is. is should contain the remainder of the input line.
 */
inline std::pair<size_t, float> read_instance_dense(std::istream &is, char &delim){
	std::pair<size_t,float> result(1,0.0f);

	float val;
	string chunk;
	while(is.good()){
		getline(is,chunk,delim);
		val=strtof(chunk.c_str(),NULL);
		chunk.clear();
		++result.first;
		result.second+=val*val;
	}

	return result;
}

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Analyses the specified data set and returns some key properties:\n"
			"- dim: number of input dimensions (higher means slower approximation),\n"
			"- maxnorm: maximum squared norm over all data instances,\n"
			"- maxgamma: maximum allowed value for gamma to remain within approximation bounds.\n"
			"  maxgamma = 1/maxnorm*sqrt(1/8)"
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
	description="data file";
	CLI::Argument<string> datafname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&datafname);

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
	description = "-> ignores the first column of the data set";
	multilinedesc.push_back(description);
	keyword = "-truth";
	CLI::FlagArgument labeled(multilinedesc,keyword,false);
	allargs.push_back(&labeled);

	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(!(datafname))
		exit_with_help(allargs,helpheader,helpfooter,false);

	std::ifstream ifile(datafname[0].c_str());

	char delimiter=*" ";
	if(dense.value())
		delimiter=delim[0];

	string line;
	std::istringstream iss;

	size_t d=0;
	float maxnorm=0.0f;

	std::pair<size_t,float> stats;

	string label;
	while(ifile.good()){
		line.clear();
		getline(ifile,line);
		if(line.empty()) break;

		iss.clear();
		iss.str(line);

		// read label
		if(labeled.value()){
			label.clear();
			getline(iss,label,delimiter);
		}

		stats = dense.value() ? read_instance_dense(iss,delimiter) : stats=read_instance_sparse(iss);
		d = (d > stats.first) ? d : stats.first;
		maxnorm = (maxnorm > stats.second) ? maxnorm : stats.second;

	}
	ifile.close();

	std::cout << "dim " << d << std::endl;
	std::cout << "maxnorm " << maxnorm << std::endl;
//	float maxgamma=1/maxnorm*sqrt(0.125);
	float maxgamma=0.25/maxnorm;
	std::cout << "maxgamma " << maxgamma << std::endl;

	return EXIT_SUCCESS;
}
