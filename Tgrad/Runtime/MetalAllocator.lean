/-! # Tgrad.Runtime.MetalAllocator — MTLBuffer alloc/free + LRU pool

  Lift from theograd_phases/13_buffer_allocator/Demo.lean (G1).
  Binds to the C bridge at c/metal_alloc_lean.c which wraps
  c/metal_alloc.m's Objective-C MTLBuffer alloc/free + an
  LRU pool. All bindings are `IO`-wrapped to prevent Lean's
  compiler from CSE-ing two same-size alloc calls into one.
-/
namespace Tgrad

namespace Runtime

namespace Metal

/-- Allocate a Metal buffer of `size` bytes. Returns 0 on failure
    (no device / OOM) or the buffer's raw pointer as a `UInt64`. -/
@[extern "lean_theograd_metal_alloc"]
opaque metalAlloc (size : USize) : IO UInt64

/-- Return a buffer to the LRU pool (or release it if pool is full). -/
@[extern "lean_theograd_metal_free"]
opaque metalFree (ptr : UInt64) (size : USize) : IO Unit

/-- Number of bytes in an MTLBuffer (sanity check). -/
@[extern "lean_theograd_metal_buffer_length"]
opaque metalBufferLength (ptr : UInt64) : IO USize

/-- Number of buffers currently parked in the LRU. -/
@[extern "lean_theograd_metal_lru_count"]
opaque metalLruCount : IO UInt32

/-- Release every cached buffer. -/
@[extern "lean_theograd_metal_lru_flush"]
opaque metalLruFlush : IO Unit

end Metal

end Runtime

end Tgrad
