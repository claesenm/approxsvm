#       -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
AC_PREREQ([2.67])
AC_INIT([ApproxSVM], [1.0], [marc.claesen@esat.kuleuven.be],
[approxsvm],[http://homes.esat.kuleuven.be/~claesenm/approxsvm/]) 
AC_CONFIG_SRCDIR([src/CLI.cpp])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE([foreign 1.5 no-define dist-bzip2 -Wall -Wno-portability subdir-objects])

#####
# BASIC CONFIGURATION
#####

# todo: make "-optimize" flag to enable "-O3 -march=native" and ATLAS/BLAS when available.
# todo: use static linking with -optimize!

####
# ATLAS: requires manual activation using --with-atlas
####

AC_CHECK_LIB([pthread],[main],[LIBS="-lpthread $LIBS"],[pthread_found=no])

AC_ARG_WITH([atlas],
  [AS_HELP_STRING([--with-atlas],
    [use ATLAS for matrix operations (overrides BLAS) @<:@default=no@:>@])],
  [LIBS="-latlas -lpthread $LIBS"],
  [with_atlas=no])

# todo: check for libpthread -> ptcblas in atlas
#[AC_SEARCH_LIBS([cblas_sdot], [ptcblas cblas],

LIBATLAS=
AS_IF([test "x$with_atlas" != xno], 
	[AC_SEARCH_LIBS([cblas_sdot], [cblas],
    	[AC_DEFINE([HAVE_LIBATLAS], [1], [Define if you have libatlas])],
    	[AC_MSG_FAILURE([--with-atlas was given, but test for atlas failed])],
   	[-latlas])
])

#LIBATLAS=
#AS_IF([test "x$with_atlas" != xno], 
#	[AC_CHECK_LIB([ptf77blas], [sdot_],
#    	[AC_DEFINE([HAVE_LIBATLAS], [1], [Define if you have libatlas])],
#    	[AC_MSG_FAILURE([--with-atlas was given, but test for atlas failed])],
#   	[])
#])


#LIBATLAS=
#AS_IF([test "x$with_atlas" != xno], 
#	[AC_CHECK_LIB([atlas], [ATL_sdot],
#    	[LIBS="-latlas $LIBS" #AC_SUBST([LIBATLAS], ["-latlas"])
#    	AC_DEFINE([HAVE_LIBATLAS], [1], [Define if you have libatlas])],
#    	[AC_MSG_FAILURE([--with-atlas was given, but test for atlas failed])],
#  	[])
#])


####
# BLAS: activated by default (when available) if ATLAS is not actived
####

AC_ARG_WITH([blas],
  [AS_HELP_STRING([--with-blas],
    [use BLAS for matrix operations @<:@default=check@:>@])],
  [],
  [with_blas=check])

#AS_IF([test "x$with_blas" != xno],
LIBBLAS=
AS_IF([test "x$with_atlas" == xno && test "x$with_blas" != xno],
  [AC_CHECK_LIB([blas], [sdot_], 
  	[LIBS="-lblas $LIBS" #AC_SUBST([LIBBLAS], ["-lblas"])
     AC_DEFINE([HAVE_LIBBLAS], [1], [Define if you have libblas])],
    [if test "x$with_blas" != xcheck; then
       AC_MSG_FAILURE([--with-blas was given, but test for blas failed])
     fi],
     [])
])

# Library version, this is an LDFLAG for Libtool -version-info 0:0:0
# http://www.sourceware.org/autobook/autobook/autobook_91.html#SEC91
AC_SUBST([SHLIBVER],[0:0:0])

AC_SUBST([PACKAGE_VERSION])

####
# LIBTOOL STUFF
####
LT_INIT
AC_LIBOBJ


# Checks for programs.
AC_PROG_CC
AC_PROG_CXX
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

AC_FUNC_MALLOC
AC_FUNC_REALLOC
AC_FUNC_STRTOD

# Checks for header files.
AC_CHECK_HEADERS([float.h limits.h locale.h stdlib.h string.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_HEADER_STDBOOL
AC_C_INLINE

# Check if we are working on Darwin
OS=shell uname
AM_CONDITIONAL(COMPILING_ON_DARWIN, [test x"$OS" = xDarwin])

# Extra flags
#AC_ARG_ENABLE([debug], AS_HELP_STRING([--enable-debug], [turn on debugging]),
#[case "${enableval}" in
#  yes) debug=true ;;
#  no)  debug=false ;;
#  *) AC_MSG_ERROR([bad value ${enableval} for --enable-debug]) ;;
#esac],[debug=false])
#AM_CONDITIONAL([DEBUG], [test x$debug = xtrue])

#if test -z "$CXXFLAGS"; then
#	CXXFLAGS=-O2
#fi

# Checks for library functions.
AC_CHECK_FUNCS([pow exp strtol strtoul strtof])

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_FILES([Makefile])

# http://www.sourceware.org/autobook/autobook/autobook_86.html#SEC86
Xsed="sed -e s/^X//"
LTLIBOBJS=`echo X"$LIBOBJS"|\
 [$Xsed -e "s,\.[^.]* ,.lo ,g;s,\.[^.]*$,.lo,"]`
AC_SUBST(LTLIBOBJS)

AC_OUTPUT
