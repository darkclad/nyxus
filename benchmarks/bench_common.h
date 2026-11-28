#pragma once

// Helpers shared by the benchmark executables.

#include <algorithm>
#include <cmath>
#include <vector>

namespace bench
{
inline double median (std::vector<double> v)
{
	std::sort (v.begin(), v.end());
	const size_t n = v.size();
	return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Median absolute deviation from the median: the spread a comparison of two medians needs to tell
// a change from noise.
inline double mad (const std::vector<double>& v)
{
	const double m = median (v);
	std::vector<double> d;
	d.reserve (v.size());
	for (double x : v)
		d.push_back (std::fabs (x - m));
	return median (d);
}
}	// namespace bench
