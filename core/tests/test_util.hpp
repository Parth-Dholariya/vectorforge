#pragma once

// A ~40 line test harness instead of GoogleTest. The dependency isn't worth it
// yet: FetchContent needs network access on every clean build, which makes CI
// flaky for no gain while the suite is this small. Swap it for gtest when the
// suite needs fixtures or parameterised cases.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace vftest {

inline int g_failures = 0;
inline int g_checks = 0;

inline void report(bool ok, const std::string& expr, const char* file, int line,
                   const std::string& detail = "") {
    ++g_checks;
    if (ok) return;
    ++g_failures;
    std::printf("  FAIL %s:%d  %s%s%s\n", file, line, expr.c_str(),
                detail.empty() ? "" : "  -> ", detail.c_str());
}

inline int summary(const char* suite) {
    std::printf("%s: %d checks, %d failures\n", suite, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}

}  // namespace vftest

#define CHECK(cond) ::vftest::report((cond), #cond, __FILE__, __LINE__)

#define CHECK_NEAR(a, b, tol)                                                     \
    ::vftest::report(std::fabs((a) - (b)) <= (tol), #a " ~= " #b, __FILE__,        \
                     __LINE__,                                                     \
                     "got " + std::to_string(a) + " want " + std::to_string(b))

#define CHECK_GE(a, b)                                                            \
    ::vftest::report((a) >= (b), #a " >= " #b, __FILE__, __LINE__,                 \
                     "got " + std::to_string(a) + " vs " + std::to_string(b))

#define CHECK_EQ(a, b)                                                            \
    ::vftest::report((a) == (b), #a " == " #b, __FILE__, __LINE__,                 \
                     "got " + std::to_string(a) + " vs " + std::to_string(b))
