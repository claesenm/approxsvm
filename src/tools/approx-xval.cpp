/**
 *  Copyright (C) 2014 Marc Claesen
 *
 *  This file is part of ApproxSVM.
 *
 * approx-xval.cpp
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
#include "svm.h"
#include <memory>
#include "Approximation.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

using std::string;
using namespace approx;

std::string toolname("approx-xval");

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

/**
 * Places content of xnode in preallocated xvec.
 *
 * If xvec is too small, we travel to UB-land.
 */
void convert(svm_node *xnode, std::vector<float> &xvec){
	int lastidx=1, k=0;
	while(xnode[k].index != -1){
		while(xnode[k].index > lastidx){
			xvec[lastidx++-1]=0.0f;
		}
		xvec[lastidx++-1]=xnode[k++].value;
	}
	while(lastidx<xvec.size())
		xvec[lastidx++]=0.0f;
}

/**
 * BEGIN OF SHAMELESS COPY FROM LIBSVM'S svm-train.c
 * To ensure file IO is identical.
 */

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

struct svm_parameter param;		// set by parse_command_line
static int max_line_len;
static char *line = NULL;
struct svm_problem prob;		// set by read_problem
struct svm_model *model;
struct svm_node *x_space;

static char* readline(FILE *input)
{
	int len;

	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}


int read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
	return max_index;
}

// end of shameless LIBSVM copy

/**
 * BEGIN OF SHAMELESS COPY FROM LIBSVM'S svm.c
 * To ensure cross-validation procedure is exactly as in LIBSVM, bar the prediction step.
 */
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void svm_group_classes(const svm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set.
	// However, for two-class sets with -1/+1 labels and -1 appears first,
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}

	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

void cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target, int max_index){

	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	//	if((param->svm_type == C_SVC ||
	//			param->svm_type == NU_SVC) && nr_fold < l)
	//	{
	int *start = NULL;
	int *label = NULL;
	int *count = NULL;
	svm_group_classes(prob,&nr_class,&label,&start,&count,perm);

	// fixme: not working correctly for prob.l <= nr_fold

	// random shuffle and then data grouped by fold using the array perm
	int *fold_count = Malloc(int,nr_fold);
	int c;
	int *index = Malloc(int,l);
	for(i=0;i<l;i++)
		index[i]=perm[i];
	for (c=0; c<nr_class; c++)
		for(i=0;i<count[c];i++)
		{
			int j = i+rand()%(count[c]-i);
			swap(index[start[c]+j],index[start[c]+i]);
		}
	for(i=0;i<nr_fold;i++)
	{
		fold_count[i] = 0;
		for (c=0; c<nr_class;c++)
			fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
	}
	fold_start[0]=0;
	for (i=1;i<=nr_fold;i++)
		fold_start[i] = fold_start[i-1]+fold_count[i-1];
	for (c=0; c<nr_class;c++)
		for(i=0;i<nr_fold;i++)
		{
			int begin = start[c]+i*count[c]/nr_fold;
			int end = start[c]+(i+1)*count[c]/nr_fold;
			for(int j=begin;j<end;j++)
			{
				perm[fold_start[i]] = index[j];
				fold_start[i]++;
			}
		}
	fold_start[0]=0;
	for (i=1;i<=nr_fold;i++)
		fold_start[i] = fold_start[i-1]+fold_count[i-1];
	free(start);
	free(label);
	free(count);
	free(index);
	free(fold_count);
	//	}

	std::vector<float> z(max_index);
	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct svm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct svm_node*,subprob.l);
		subprob.y = Malloc(double,subprob.l);

		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct svm_model *submodel = svm_train(&subprob,param);
		std::auto_ptr<approx::Model> approx=approx::Model::approximate(*submodel);

//		double dec_val=0.0;


		if(approx::Classifier *classifier=dynamic_cast<approx::Classifier*>(approx.get())){
			for(j=begin;j<end;j++){
				// target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
				convert(prob->x[perm[j]],z);
				target[perm[j]]=classifier->classify(z).first;

//				svm_predict_values(submodel,prob->x[perm[j]],&dec_val);
//				std::cout << classifier->classify(z).second << " " << dec_val << std::endl;
			}
		}else{

			for(j=begin;j<end;j++){
				// target[perm[j]] = svm_predict(submodel,prob->x[perm[j]]);
				convert(prob->x[perm[j]],z);
				target[perm[j]]=approx->evaluate(z);
			}
		}
		svm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}
	free(fold_start);
	free(perm);

}

// end of shameless LIBSVM copy 2

int main(int argc, char **argv)
{
	// initialize help
	std::string helpheader(
			"Performs k-fold cross-validation using the RBF kernel approximation in prediction.\n"
			"\n"
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
	description="data file (in LIBSVM format)";
	CLI::Argument<string> datafname(description,keyword,CLI::Argument<string>::Content(1,""));
	allargs.push_back(&datafname);

	keyword="-c";
	description="set the parameter C of C-SVC";
	CLI::Argument<double> c(description,keyword,CLI::Argument<double>::Content(1,1.0));
	allargs.push_back(&c);

	keyword="-g";
	description="set gamma in the kernel function";
	CLI::Argument<double> gamma(description,keyword,CLI::Argument<double>::Content(1,1.0));
	allargs.push_back(&gamma);

	keyword="-m";
	description="set LIBSVM cache memory size in MB (default 100)";
	CLI::Argument<double> cache(description,keyword,CLI::Argument<double>::Content(1,100.0));
	allargs.push_back(&cache);

	keyword="-h";
	description="shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)";
	CLI::Argument<unsigned> shrinking(description,keyword,CLI::Argument<unsigned>::Content(1,1));
	allargs.push_back(&shrinking);

	keyword="-v";
	description="set k of k-fold cross-validation mode (default 5)";
	CLI::Argument<int> nfolds(description,keyword,CLI::Argument<int>::Content(1,5));
	allargs.push_back(&nfolds);


	ParseCLI(argv,argc,1,allargs);

	if(help.configured() || help2.configured())
		exit_with_help(allargs,helpheader,helpfooter,true);
	if(version.configured() || version2.configured())
		exit_with_version(toolname);

	if(!(datafname))
		exit_with_help(allargs,helpheader,helpfooter,false);

	if(gamma[0] == 0) exit_with_err("Illegal value for gamma (0), please see -g.");
	if(cache[0] == 0) exit_with_err("Illegal value for cache size (0), please see -m.");
	if(nfolds[0] < 2) exit_with_err("Illegal value for number of folds (0), please see -v.");

	//	static double weights[]={1.0, 1.0};

	int max_index=read_problem(datafname[0].c_str());

	param.svm_type = C_SVC;
	param.kernel_type = RBF;
	param.degree = 3;
	param.gamma = 0;	// 1/num_features
	param.coef0 = 0;
	param.nu = 0.5;
	param.cache_size = 100;
	param.C = 1;
	param.eps = 1e-3;
	param.p = 0.1;
	param.shrinking = 1;
	param.probability = 0;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;

	param.C=c[0];
	param.gamma=gamma[0];
	param.cache_size=cache[0];
	param.kernel_type=RBF;
	param.svm_type=C_SVC;
	param.weight_label=NULL;
	param.nr_weight=0;
	param.weight=NULL;
	param.shrinking = shrinking[0];

	double *target = Malloc(double,prob.l);
	cross_validation(&prob,&param,nfolds[0],target,max_index);

	unsigned total_correct=0;
	for(int i=0;i<prob.l;i++)
		if(target[i] == prob.y[i])
			++total_correct;

	printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	free(target);

	return EXIT_SUCCESS;
}
