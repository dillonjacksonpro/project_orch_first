// Must NOT compile: emit<Undeclared>() triggers TypeIndex<Undeclared, Declared>
// which hits the incomplete primary template — hard error when T is not in the pack.
#include "context/work_ctx.hpp"  // pulls in orchestrator_config.hpp for RING_SIZE
#include "ring/typed_ring_set.hpp"

struct Declared {};
struct Undeclared {};

void f() {
    TypedRingSet<RING_SIZE, Declared> rs;
    WorkCtx<Declared>                 ctx{0, 1, rs};
    ctx.emit(Undeclared{});
}
