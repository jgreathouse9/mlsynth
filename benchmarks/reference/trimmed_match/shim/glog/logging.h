// Stand-in for glog, for building google/trimmed_match's C++ core without Bazel.
// The core uses only CHECK(cond) << message. A failed check throws std::runtime_error, which pybind11
// turns into a Python RuntimeError, where glog would abort the process.
#pragma once
#include <sstream>
#include <stdexcept>
namespace glog_shim {
struct CheckFailure {
  std::ostringstream msg;
  explicit CheckFailure(const char* cond) { msg << "CHECK failed: " << cond << " "; }
  template <class T> CheckFailure& operator<<(const T& v) { msg << v; return *this; }
  ~CheckFailure() noexcept(false) { throw std::runtime_error(msg.str()); }
};
}  // namespace glog_shim
#define CHECK(cond) if (cond) {} else glog_shim::CheckFailure(#cond)
