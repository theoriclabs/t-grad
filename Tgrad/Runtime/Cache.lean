import Tgrad.Runtime.MetalAllocator

/-! # Tgrad.Runtime.Cache — LRU pool sentinel + flush wrapper

  L4 is a thin wrapper around the LRU pool in the C bridge. Real
  multi-size pool behaviour is exercised in the `ffi-alloc-cycle`
  gate predicate; this module exposes the controls.
-/
namespace Tgrad

namespace Runtime

namespace Cache

open Metal

/-- Drain every cached buffer. Called at the start of the L4
    alloc-cycle predicate so prior runs don't contaminate the
    pool state. -/
def flush : IO Unit := metalLruFlush

/-- Current LRU population. -/
def count : IO UInt32 := metalLruCount

end Cache

end Runtime

end Tgrad
