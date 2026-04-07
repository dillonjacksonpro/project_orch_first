// Must NOT compile: data<Undeclared>() triggers TypeIndex<Undeclared, Declared>
// which hits the incomplete primary template — hard error when T is not in the pack.
#include "context/reduce_ctx.hpp"

struct Declared   { int x; };
struct Undeclared { int y; };

void f() {
    std::vector<Declared> vd;
    ReduceCtx<Declared>   ctx{vd};
    ctx.data<Undeclared>(); // T not in Types... — compile error
}
