// Must NOT compile: brace-init without a named type cannot deduce T for emit<T>().
// The user must write ctx.emit(TypeA{42}), not ctx.emit({42}).
// Documents the error message they will see.
#include "context/work_ctx.hpp"  // pulls in orchestrator_config.hpp for RING_SIZE
#include "ring/typed_ring_set.hpp"

struct TypeA { int x; };

void f() {
    TypedRingSet<RING_SIZE, TypeA> rs;
    WorkCtx<TypeA>                 ctx{0, 1, rs};
    ctx.emit({42}); // error: cannot deduce template argument T from brace-enclosed init list
}
