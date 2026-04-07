// Must NOT compile: static_assert(AllUnique<TypeA, TypeA>::value, ...) fires
// because duplicate types in JobInterface produce ambiguous ring/buffer lookups.
#include "job_interface.hpp"

struct TypeA { int x; };

// Duplicate type — AllUnique<TypeA, TypeA>::value == false
class J : public JobInterface<TypeA, TypeA> {
public:
    void on_map(WorkCtx<TypeA, TypeA>&) override {}
};
