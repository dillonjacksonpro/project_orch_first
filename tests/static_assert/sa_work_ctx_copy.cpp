// Must NOT compile: WorkCtx copy constructor is explicitly deleted.
// WorkCtx holds a reference to thread-local storage and must fire its destructor
// exactly once in the owning OMP thread — copying is never safe.
#include "context/work_ctx.hpp"  // pulls in orchestrator_config.hpp for RING_SIZE
#include "ring/typed_ring_set.hpp"

struct TypeA { int x; };

void f() {
    TypedRingSet<RING_SIZE, TypeA> rs;
    WorkCtx<TypeA>                 ctx{0, 1, rs};
    WorkCtx<TypeA>                 copy{ctx}; // deleted copy constructor
}
