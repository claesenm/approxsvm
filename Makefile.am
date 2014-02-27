# @configure_input@

SUBDIRS = #include src/tools src/tests src libsvm

#########################
# COMPILATION FLAGS ETC #
#########################

EXTRAFLAGS=-Wall -pedantic -ffast-math

#if OPTIMIZE
#EXTRAFLAGS+=-O3 -march=native
#endif

AM_CXXFLAGS = $(EXTRAFLAGS)
AM_CPPFLAGS =-I$(top_srcdir)/include -I$(top_builddir)/libsvm

################
# DISTRIBUTION #
################

EXTRA_DIST = COPYING.LESSER LICENSE libsvm/COPYRIGHT 

#####################
# AUTOTOOLS SCRIPTS #
#####################
ACLOCAL_AMFLAGS= -I m4

include_HEADERS = include/CLI.h libsvm/svm.h include/Util.h include/MatrixMath.h include/Approximation.h

#######
# CONSTRUCT CORE LIBRARY
#######

# we include libsvm code in our shared library for svm_read()
lib_LTLIBRARIES = lib/libapproxsvm.la 

dist_lib_libapproxsvm_la_SOURCES = src/CLI.cpp libsvm/svm.cpp src/Util.cpp \
	src/MatrixMath.cpp src/Approximation.cpp 

#nodist_lib_libensemblesvm_la_SOURCES = src/libsvm/svm.cpp src/libsvm/svm.h
lib_libapproxsvm_la_LDFLAGS = -version-info $(SHLIBVER)

bin_PROGRAMS=
check_PROGRAMS=

BASELIBS=lib/libapproxsvm.la

## APPROX-SVM
bin_PROGRAMS += $(top_builddir)/bin/approx-svm
__top_builddir__bin_approx_svm_SOURCES = src/tools/approx-svm.cpp
__top_builddir__bin_approx_svm_LDADD = $(BASELIBS)

## APPROX-PREDICT
bin_PROGRAMS += $(top_builddir)/bin/approx-predict
__top_builddir__bin_approx_predict_SOURCES = src/tools/approx-predict.cpp
__top_builddir__bin_approx_predict_LDADD = $(BASELIBS)

## APPROX-ANALYSE
bin_PROGRAMS += $(top_builddir)/bin/approx-analyse
__top_builddir__bin_approx_analyse_SOURCES = src/tools/approx-analyse.cpp
__top_builddir__bin_approx_analyse_LDADD = $(BASELIBS)

## APPROX-XVAL
bin_PROGRAMS += $(top_builddir)/bin/approx-xval
__top_builddir__bin_approx_xval_SOURCES = src/tools/approx-xval.cpp
__top_builddir__bin_approx_xval_LDADD = $(BASELIBS)

## TEST BINARIES
check_PROGRAMS += $(top_builddir)/tests/matrixmath
__top_builddir__tests_matrixmath_SOURCES = src/tests/matrixmath.cpp
__top_builddir__tests_matrixmath_LDADD = $(BASELIBS)
#check_PROGRAMS += $(top_builddir)/tests/rbfapprox
#__top_builddir__tests_rbfapprox_SOURCES = src/tests/rbfapprox.cpp
#__top_builddir__tests_rbfapprox_LDADD = $(BASELIBS)
