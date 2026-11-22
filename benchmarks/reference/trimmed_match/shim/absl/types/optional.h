// Stand-in for absl/types/optional.h, for building google/trimmed_match's C++ core without Bazel.
// Under C++17 absl::optional is std::optional; the core uses optional and nullopt.
#pragma once
#include <optional>
namespace absl { using std::optional; using std::nullopt; }
