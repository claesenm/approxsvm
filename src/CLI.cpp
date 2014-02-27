/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * CLI.cpp
 *
 *      Author: Marc Claesen
 */

#include "CLI.h"
#include <vector>

namespace CLI{

unsigned ParseCLI(char **argv, unsigned argc, unsigned idx, const deque<BaseArgument*> &candidates){
	std::vector<bool> include(candidates.size(),true);
	unsigned numfiltered=0;
	bool foundcandidate;
	while(numfiltered<candidates.size() && idx<argc){
		foundcandidate=false;
		for(unsigned i=0;i<candidates.size();++i){
			if(include[i]){
				if(candidates[i]->length() > argc-idx){ // argument no longer fits in remainder of CLI arguments
					include[i]=false;
					numfiltered++;
				}else{
					pair<unsigned,bool> parseresult=candidates[i]->parse(argv,idx);// attempts to read given argument
					if(parseresult.first!=idx){ // on successful readout, filter this argument from further attempts
						include[i]=parseresult.second;
						if(!parseresult.second) numfiltered++;

						foundcandidate=true;
						idx=parseresult.first;
					}
				}
			}
		}
		if(!foundcandidate)
			// break;	// breaks parsing when nothing is found, e.g. when the supplied arguments are invalid
			++idx;	// continue parsing after receiving unknown argument
	}
	return idx;
}

void BaseArgument::printDescription(std::ostream &os) const{
	os << keyword;
	for(unsigned descidx=0;descidx<description.size();++descidx){
		if(descidx==0){
			for(unsigned i=0;i<BaseArgument::TABLENGTH-keyword.length();++i)
				os << " ";
		}else{
			for(unsigned i=0;i<BaseArgument::TABLENGTH;++i)
				os << " ";
		}
		os << description[descidx] << std::endl;
	}
}

pair<unsigned,bool> BaseArgument::parse(char **argv, unsigned idx){
	unsigned newidx=read(argv,idx);
	bool returnbool=true;;
	if(newidx!=idx){
		isconfigured=true;
		returnbool=parseAfterHit();
	}
	return std::make_pair(newidx,returnbool);
}

FlagArgument::FlagArgument(const string &description, const string &keyword, bool defvalue)
:BaseArgument(description,keyword),val(defvalue){}

FlagArgument::FlagArgument(const vector<string> &description, const string &keyword, bool defvalue)
:BaseArgument(description,keyword),val(defvalue){}

unsigned FlagArgument::read(char **argv, unsigned idx){
	string current(argv[idx]);
	if(keyword.compare(current)!=0)
		return idx;

	// if the keyword is present, swap the flagged value
	val=!val;
	return idx+1;
}
void FlagArgument::print(std::ostream &os) const{
	if(isconfigured){
		os << keyword << " (flag)";
	}else{
		BaseArgument::printDescription(os);
	}
}

SilentFlagArgument::SilentFlagArgument(const string &keyword, bool defvalue)
:FlagArgument(std::string(""),keyword,defvalue){}
void SilentFlagArgument::print(std::ostream &os) const{}

} // CLI namespace
