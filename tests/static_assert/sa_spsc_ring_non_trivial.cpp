// Must NOT compile.
// SpscRing<std::string, 16> triggers:
//   static_assert: T must be trivially copyable
#include "ring/spsc_ring.hpp"
#include <string>

SpscRing<std::string, 16> ring;
