// Must NOT compile.
// ring_set.ring<Undeclared>() triggers:
//   TypeIndex<Undeclared, Declared>: incomplete instantiation (type not in pack)
#include "ring/typed_ring_set.hpp"

struct Declared {};
struct Undeclared {};

void f() {
    TypedRingSet<16, Declared> rs;
    rs.ring<Undeclared>();  // compile error: Undeclared is not in the type pack
}
