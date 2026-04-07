// Must NOT link: on_map is declared (satisfying the abstract class requirement)
// but never defined, so the linker cannot resolve the reference.
//
// This is intentionally a LINK-fail, not a compile-fail:
//   - ConcreteJob is not abstract (on_map is declared) → compiles OK
//   - ConcreteJob::on_map has no definition                → link error
//
// Verified via the test_link_fail Makefile target (NOT test_static_assert).
#include "job_interface.hpp"

struct TypeA { int x; };

class ConcreteJob : public JobInterface<TypeA> {
public:
    void on_map(WorkCtx<TypeA>&) override; // declared but never defined
};

int main() {
    ConcreteJob job;
    WorkCtx<TypeA>* ctx = nullptr;
    job.on_map(*ctx); // forces linker to resolve ConcreteJob::on_map
    return 0;
}
