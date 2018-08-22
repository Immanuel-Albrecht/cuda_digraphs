#pragma once

/** precompiled header **/

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ratio>
#include <complex>
#include <valarray>
#include <limits>
#include <cstdint>
#include <cerrno>
#include <cassert>
#include <ctime>
#include <chrono>
#include <fstream>

#include <map>
#include <utility>
#include <algorithm>
#include <initializer_list>
#include <numeric>


#include <string>
#include <vector>
#include <set>

#include <algorithm>
#include <iterator>


#include <libpopcnt.h>

/** cuda **/

#include "cuda.h"

/* NOTE that NVCC has some problems with boost */
#ifndef __NVCC__

#include <boost/math/special_functions/binomial.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int_distribution.hpp>


#endif

#ifdef NDEBUG
#define VERBOSE(x) {}
#else
#define VERBOSE(x) { (x); }
#endif

