/-! # Tgrad.Runtime.MetalDevice — Metal device probe

  Lift from theograd_phases/13_buffer_allocator/Demo.lean. Single
  binding to `lean_theograd_metal_available` in the C bridge.

  When `metalAvailable` returns 0 on the build host, the L4+ Metal
  FFI gates cannot run; the gate runner halts at L3 rather than
  executing L4's behavioural predicates.
-/
namespace Tgrad

namespace Runtime

namespace Metal

/-- Returns 1 if `MTLCreateSystemDefaultDevice()` succeeded, 0
    otherwise. -/
@[extern "lean_theograd_metal_available"]
opaque metalAvailable : IO UInt8

end Metal

end Runtime

end Tgrad
