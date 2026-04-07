// Must NOT compile: TypeIndex<Undeclared, Declared> hits the incomplete primary
// template, producing a hard error when T is not in the TypedLocalBuffer pack.
#include "buffer/typed_local_buffer.hpp"
#include "ring/typed_ring_set.hpp"

struct Declared {};
struct Undeclared {};

void f() {
    TypedRingSet<16, Declared>      rs;
    TypedLocalBuffer<4, 16, Declared> buf;
    buf.write(Undeclared{}, rs);
}
