// Kernel-only timing of the 2D feature classes that dominate a featurization run.
//
// Each kernel runs FeatureMethod::calculate() on one synthetic ROI, with no image loading, no ROI
// gathering and no thread pool, so a change in the number is a change in the kernel. The ROI is a
// blob of the given side whose radius is modulated by two harmonics (non-convex, like the ROIs of
// benchmarks/perf), over a seeded texture. The blob's pixels are generated once per side, and the
// features a kernel depends on (the contour and the convex hull) once per kernel and side; each
// repetition gets a fresh ROI built from those, so a kernel that changes its ROI cannot change the
// next repetition's input. None of that setup is timed.
//
// Output is CSV: kernel, side, ROI pixels, repetitions, median/MAD/min/max milliseconds, and a
// value the kernel produced, so a timing change that comes with a value change is visible.
// --reps and --no-warmup fix the number of runs, and --no-timing leaves the clock out and runs each
// repetition through bench_repetition(), whose kernel call is bench_kernel(), so an instruction
// counter (benchmarks/perf, `icount`) can collect inside that function alone and get one kernel
// call's instructions, the same every time.

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

#include "../src/nyx/environment.h"
#include "../src/nyx/globals.h"
#include "../src/nyx/roi_cache.h"
#include "../src/nyx/features/2d_geomoments.h"
#include "../src/nyx/features/contour.h"
#include "../src/nyx/features/convex_hull.h"
#include "../src/nyx/features/erosion.h"
#include "../src/nyx/features/gabor.h"
#include "../src/nyx/features/glcm.h"
#include "../src/nyx/features/gldzm.h"
#include "../src/nyx/features/glrlm.h"
#include "../src/nyx/features/glszm.h"
#include "../src/nyx/features/ngldm.h"
#include "../src/nyx/features/zernike.h"
#include "bench_common.h"

namespace
{
using Clock = std::chrono::steady_clock;

// The pixels of one blob ROI, side x side bounding box, intensities 100..60000.
std::vector<Pixel2> blob_pixels (int side, unsigned seed)
{
	std::mt19937 rng (seed);
	std::uniform_real_distribution<double> phase (0.0, 6.283185307179586);
	std::normal_distribution<double> noise (0.0, 0.08);
	const double a = phase (rng), b = phase (rng), c = (side - 1) / 2.0, r0 = 0.42 * side;

	std::vector<Pixel2> px;
	for (int y = 0; y < side; y++)
		for (int x = 0; x < side; x++)
		{
			const double dx = x - c, dy = y - c, th = std::atan2 (dy, dx);
			if (std::hypot (dx, dy) > r0 * (1.0 + 0.22 * std::sin (3 * th + a) + 0.08 * std::sin (7 * th + b)))
				continue;
			double v = 0.5 + 0.5 * (0.35 * std::sin (6.283185307179586 * (3.0 * x + 2.0 * y) / side)
				+ 0.25 * std::cos (6.283185307179586 * (7.0 * y - 5.0 * x) / side)) / 0.6 + noise (rng);
			v = std::clamp (v, 0.0, 1.0);
			px.push_back (Pixel2 (x, y, static_cast<PixIntens> (100 + v * 59900)));
		}
	return px;
}

// An ROI holding the given pixels, with its bounding box and image matrix.
void make_roi (LR& r, const std::vector<Pixel2>& px)
{
	r.initialize_fvals();
	for (size_t i = 0; i < px.size(); i++)
		if (i == 0)
			Nyxus::init_label_record_3 (r, (int) px[i].x, (int) px[i].y, (PixIntens) px[i].inten);
		else
			Nyxus::update_label_record_3 (r, (int) px[i].x, (int) px[i].y, (PixIntens) px[i].inten);
	r.raw_pixels = px;
	r.make_nonanisotropic_aabb();
	r.aux_image_matrix.allocate (r.aabb.get_width(), r.aabb.get_height());
	r.aux_image_matrix.calculate_from_pixelcloud (r.raw_pixels, r.aabb);
}

// What ContourFeature and ConvexHullFeature leave on an ROI, so it can be given to a fresh one.
struct Dependencies
{
	std::vector<std::vector<Pixel2>> multicontour;
	std::vector<bool> multicontour_hole;
	std::vector<Pixel2> contour_pixels, convex_hull;
	std::vector<size_t> contour_border_ends;
	std::vector<std::vector<StatsReal>> fvals;

	void take (const LR& r)
	{
		multicontour = r.multicontour_;
		multicontour_hole = r.multicontour_hole_;
		contour_pixels = r.contour_pixels_;
		contour_border_ends = r.contour_border_ends_;
		convex_hull = r.convHull_CH;
		fvals = r.fvals;
	}
	void give (LR& r) const
	{
		r.multicontour_ = multicontour;
		r.multicontour_hole_ = multicontour_hole;
		r.contour_pixels_ = contour_pixels;
		r.contour_border_ends_ = contour_border_ends;
		r.convHull_CH = convex_hull;
		r.fvals = fvals;
	}
};

struct Kernel
{
	const char* name;
	std::function<void (LR&, Environment&)> prepare;	// untimed: dependencies
	std::function<double (LR&, Environment&)> run;		// timed; returns one produced value
	int max_side;										// skip larger sides (cost grows too fast)
};

template <class F>
double calc (LR& r, Environment& e, Nyxus::Feature2D probe)
{
	F f;
	f.calculate (r, e.get_feature_settings (typeid (F)));
	f.save_value (r.fvals);
	return r.fvals[(int) probe][0];
}

void contour_and_hull (LR& r, Environment& e)
{
	calc<ContourFeature> (r, e, Nyxus::Feature2D::PERIMETER);
	calc<ConvexHullFeature> (r, e, Nyxus::Feature2D::CONVEX_HULL_AREA);
}

std::vector<Kernel> kernels()
{
	auto none = [] (LR&, Environment&) {};
	using F = Nyxus::Feature2D;
	return {
		{ "contour", none, [] (LR& r, Environment& e) { return calc<ContourFeature> (r, e, F::PERIMETER); }, 4096 },
		{ "convex_hull", [] (LR& r, Environment& e) { calc<ContourFeature> (r, e, F::PERIMETER); },
			[] (LR& r, Environment& e) { return calc<ConvexHullFeature> (r, e, F::CONVEX_HULL_AREA); }, 4096 },
		{ "erosion", contour_and_hull, [] (LR& r, Environment& e) { return calc<ErosionPixelsFeature> (r, e, F::EROSIONS_2_VANISH); }, 1024 },
		{ "gabor", none, [] (LR& r, Environment& e) { return calc<GaborFeature> (r, e, F::GABOR); }, 1024 },
		{ "imoments", contour_and_hull, [] (LR& r, Environment& e) { return calc<Imoms2D_feature> (r, e, F::IMOM_RM_00); }, 2048 },
		{ "smoments", contour_and_hull, [] (LR& r, Environment& e) { return calc<Smoms2D_feature> (r, e, F::SPAT_MOMENT_00); }, 2048 },
		{ "zernike", none, [] (LR& r, Environment& e) { return calc<ZernikeFeature> (r, e, F::ZERNIKE2D); }, 1024 },
		{ "glcm", none, [] (LR& r, Environment& e) { return calc<GLCMFeature> (r, e, F::GLCM_CONTRAST); }, 4096 },
		{ "glrlm", none, [] (LR& r, Environment& e) { return calc<GLRLMFeature> (r, e, F::GLRLM_RLN); }, 4096 },
		{ "glszm", none, [] (LR& r, Environment& e) { return calc<GLSZMFeature> (r, e, F::GLSZM_SZN); }, 4096 },
		{ "gldzm", none, [] (LR& r, Environment& e) { return calc<GLDZMFeature> (r, e, F::GLDZM_SDE); }, 4096 },
		{ "ngldm", none, [] (LR& r, Environment& e) { return calc<NGLDMfeature> (r, e, F::NGLDM_LDE); }, 4096 },
	};
}

using bench::mad;
using bench::median;

int reps_for (int side)
{
	return side <= 64 ? 50 : side <= 256 ? 15 : side <= 1024 ? 5 : 3;
}

#ifndef NYXUS_BENCH_BUILD_TYPE	// set by CMakeLists.txt
#define NYXUS_BENCH_BUILD_TYPE "unknown"
#endif

std::string compiler_id()
{
#if defined(__clang__)
	return std::string ("clang ") + __clang_version__;
#elif defined(__GNUC__)
	return "gcc " + std::to_string (__GNUC__) + "." + std::to_string (__GNUC_MINOR__) + "." + std::to_string (__GNUC_PATCHLEVEL__);
#elif defined(_MSC_FULL_VER)
	return "msvc " + std::to_string (_MSC_FULL_VER);
#else
	return "unknown";
#endif
}

int parse_int (std::string_view s, bool positive)
{
	const std::string t (s);
	char* end = nullptr;
	errno = 0;
	const long long v = std::strtoll (t.c_str(), &end, 10);
	if (t.empty() || !end || *end)
		throw std::runtime_error ("expected an integer, got " + t);
	if (errno == ERANGE || v < std::numeric_limits<int>::min() || v > std::numeric_limits<int>::max())
		throw std::runtime_error ("integer out of range, got " + t);
	if (positive && v <= 0)
		throw std::runtime_error ("expected a positive integer, got " + t);
	return (int) v;
}
}	// namespace

// A repetition's input: a fresh ROI from the cached pixels with the prepared dependencies. Both the
// timed loop and bench_repetition() build it here, so timings and instruction counts measure the
// same repetition.
static void prepare_repetition (LR& r, const std::vector<Pixel2>& px, Dependencies& deps)
{
	make_roi (r, px);
	deps.give (r);
}

#if defined(__GNUC__)
#define BENCH_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define BENCH_NOINLINE __declspec(noinline)
#else
#define BENCH_NOINLINE
#endif

// The kernel alone, on a prepared ROI. Not inlined and not in a namespace, so an instruction
// counter can collect inside it and nothing else (callgrind --toggle-collect=bench_kernel*): the
// count is the kernel's, not the ROI's preparation, whose cost grows with every feature code Nyxus
// registers (it sizes the ROI's feature-value table).
BENCH_NOINLINE double bench_kernel (const Kernel& k, LR& r, Environment& env)
{
	return k.run (r, env);
}

// One repetition, untimed: prepare_repetition(), then bench_kernel().
double bench_repetition (const Kernel& k, const std::vector<Pixel2>& px, Dependencies& deps, Environment& env)
{
	LR r (1);
	prepare_repetition (r, px, deps);
	return bench_kernel (k, r, env);
}

int main (int argc, char** argv)
{
	std::vector<int> sides;
	std::vector<std::string> only;
	int greydepth = 64;
	int reps_override = 0;	// 0: reps_for(side)
	bool warmup = true;
	bool timing = true;	// false: no clock reads, times printed as 0, so a run's instructions repeat exactly
	try
	{
		for (int i = 1; i < argc; i++)
		{
			const std::string_view a (argv[i]);
			auto next = [&] () -> std::string_view {
				if (++i >= argc) throw std::runtime_error (std::string (a) + " needs a value");
				return argv[i];
			};
			if (a == "--side") sides.push_back (parse_int (next(), true));
			else if (a == "--kernel") only.emplace_back (next());
			// any coarse grey depth Nyxus takes: N > 0 bins, 0 does not bin, N < 0 is |N| PyRadiomics-style bins
			else if (a == "--greydepth") greydepth = parse_int (next(), false);
			else if (a == "--reps") reps_override = parse_int (next(), true);
			else if (a == "--no-warmup") warmup = false;
			else if (a == "--no-timing") timing = false;
			else if (a == "--build-info")
			{
				std::cout << "compiler " << compiler_id() << "\n" << "pointer_bits " << 8 * sizeof (void*) << "\n"
					<< "build_type " << NYXUS_BENCH_BUILD_TYPE << "\n"
#ifdef USE_GPU
					<< "gpu on\n"
#else
					<< "gpu off\n"
#endif
#ifdef CHECKTIMING
					<< "checktiming on\n";
#else
					<< "checktiming off\n";
#endif
				return 0;
			}
			else if (a == "--help" || a == "-h")
			{
				std::cout << "Usage: " << argv[0]
					<< " [--side N ...] [--kernel NAME ...] [--greydepth N] [--reps N] [--no-warmup] [--no-timing]"
					" [--build-info]\n"
					<< "Each kernel runs at the given sides up to its largest side, listed after its name.\n"
					<< "--greydepth is the coarse grey depth of the texture kernels (default 64): N > 0 bins to N\n"
					<< "levels, 0 does not bin, N < 0 bins to |N| PyRadiomics-style bins.\n"
					<< "--reps sets the timed repetitions (default by side), --no-warmup drops the untimed first\n"
					<< "run, --no-timing reads no clock and prints 0 times, and --build-info prints the compiler.\n"
					<< "An instruction counter runs a kernel with --reps 1 --no-warmup --no-timing and collects\n"
					<< "inside bench_kernel() only: the clock reads, and printing times that differ from\n"
					<< "run to run, would make the count differ.\n"
					<< "Kernels:\n";
				for (const auto& k : kernels()) std::cout << "  " << k.name << ' ' << k.max_side << "\n";
				return 0;
			}
			else throw std::runtime_error ("unknown argument " + std::string (a));
		}
		const auto all = kernels();
		for (const auto& name : only)
			if (std::none_of (all.begin(), all.end(), [&] (const Kernel& k) { return name == k.name; }))
				throw std::runtime_error ("unknown kernel " + name + " (--help lists them)");
	}
	catch (const std::exception& ex)
	{
		std::cerr << "bench_feature_kernels: " << ex.what() << "\n";
		return 1;
	}
	if (sides.empty())
		sides = { 64, 128, 256, 512, 1024, 2048 };

	Environment env;
	env.set_coarse_gray_depth (greydepth);
	env.compile_feature_settings();

	std::map<int, std::vector<Pixel2>> blobs;	// side -> pixels, generated on first use
	std::cout << "kernel,side,pixels,reps,median_ms,mad_ms,min_ms,max_ms,value\n";
	for (const auto& k : kernels())
	{
		if (!only.empty() && std::find (only.begin(), only.end(), k.name) == only.end())
			continue;
		for (int side : sides)
		{
			if (side > k.max_side)
				continue;
			const int reps = reps_override > 0 ? reps_override : reps_for (side);
			const int first_timed = warmup ? 1 : 0;	// the first run warms caches and allocators
			auto blob = blobs.find (side);
			if (blob == blobs.end())
				blob = blobs.emplace (side, blob_pixels (side, 12345u + (unsigned) side)).first;
			const std::vector<Pixel2>& px = blob->second;
			Dependencies deps;
			{
				LR r (1);
				make_roi (r, px);
				k.prepare (r, env);
				deps.take (r);
			}
			std::vector<double> ms;
			double value = 0;
			const size_t npix = px.size();
			for (int i = 0; i < reps + first_timed; i++)
			{
				if (!timing)
				{
					value = bench_repetition (k, px, deps, env);
					if (i >= first_timed)
						ms.push_back (0.0);
					continue;
				}
				LR r (1);
				prepare_repetition (r, px, deps);
				const auto t0 = Clock::now();	// the kernel alone: preparing its input is not timed
				value = k.run (r, env);
				const auto t1 = Clock::now();
				if (i >= first_timed)
					ms.push_back (std::chrono::duration<double, std::milli> (t1 - t0).count());
			}
			std::cout << k.name << ',' << side << ',' << npix << ',' << reps << ','
				<< std::fixed << std::setprecision (4) << median (ms) << ',' << mad (ms) << ','
				<< *std::min_element (ms.begin(), ms.end()) << ',' << *std::max_element (ms.begin(), ms.end()) << ','
				// every significant digit: a value change of any size must show in the CSV
				<< std::defaultfloat << std::setprecision (17) << value << '\n' << std::flush;
		}
	}
	return 0;
}
