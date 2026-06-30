/-! # Tgrad.Runtime.MetalProgram — MSL compile + kernel dispatch

  Lift from theograd_phases/14_dispatch_executor/Demo.lean (G2).
  Binds to the C bridge for MSL compile + dispatch.
-/
namespace Tgrad

namespace Runtime

namespace Metal

/-- Compile MSL source. Returns the MTLLibrary pointer (as UInt64) on
    success, 0 on failure. -/
@[extern "lean_theograd_metal_compile"]
opaque metalCompile (msl : @& String) : IO UInt64

/-- Release a library returned by `metalCompile`. -/
@[extern "lean_theograd_metal_library_release"]
opaque metalLibraryRelease (libPtr : UInt64) : IO Unit

/-- Number of `[[kernel]]` functions in the compiled library. -/
@[extern "lean_theograd_metal_library_function_count"]
opaque metalLibraryFunctionCount (libPtr : UInt64) : IO UInt32

/-- Synchronous kernel dispatch. Returns 0 on success, negative on
    failure (see metal_alloc.m for the error-code mapping; NULL
    buffer pointer surfaces as a negative rc). -/
@[extern "lean_theograd_metal_dispatch"]
opaque metalDispatch
    (libPtr : UInt64) (fnName : @& String) (buffers : @& Array UInt64)
    (gx gy gz lx ly lz : USize) : IO UInt32

end Metal

end Runtime

end Tgrad
