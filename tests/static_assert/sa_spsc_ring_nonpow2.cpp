// Must NOT compile.
// SpscRing<int, 15> triggers:
//   static_assert: N must be a power of 2
#include "ring/spsc_ring.hpp"

SpscRing<int, 15> ring;
