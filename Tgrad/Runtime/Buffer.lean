import Tgrad.Runtime.MetalAllocator

/-! # Tgrad.Runtime.Buffer — typed BufferHandle + host I/O

  Wraps the raw `UInt64` MTLBuffer pointer in a typed struct + binds
  the host read/write helpers. Lift from theograd_phases/13/14.
-/
namespace Tgrad

namespace Runtime

/-- Opaque-handle wrapper around an MTLBuffer pointer. The raw value
    is the `UInt64` MTLBuffer address returned by `metalAlloc`. -/
structure BufferHandle where
  raw  : UInt64
  size : Nat
  deriving Repr, Inhabited

namespace Metal

/-- Write a FloatArray to a buffer's contents (host → device, f32). -/
@[extern "lean_theograd_metal_buffer_write_f32"]
opaque metalBufferWriteF32 (bufPtr : UInt64) (vals : @& FloatArray) : IO Unit

/-- Read a single f32 element from a buffer's contents at index `i`. -/
@[extern "lean_theograd_metal_buffer_read_f32"]
opaque metalBufferReadF32 (bufPtr : UInt64) (index : USize) : IO Float

/-- Copy raw bytes from a Lean `ByteArray` into the buffer's contents at
    offset 0. Used for arbitrary-dtype host→device transfer (notably
    bf16, which has no native numpy dtype). L5.b uses this for the
    byte-match-vs-captured-tinygrad predicate. -/
@[extern "lean_theograd_metal_buffer_write_bytes"]
opaque metalBufferWriteBytes (bufPtr : UInt64) (bytes : @& ByteArray) : IO Unit

/-- Copy `nBytes` from the buffer's contents into a fresh `ByteArray`.
    Symmetric counterpart of `metalBufferWriteBytes`. -/
@[extern "lean_theograd_metal_buffer_read_bytes"]
opaque metalBufferReadBytes (bufPtr : UInt64) (nBytes : USize) : IO ByteArray

end Metal

namespace Buffer

/-- Allocate a typed buffer of `n` f32 elements. -/
def allocF32 (n : Nat) : IO BufferHandle := do
  let bytes : USize := USize.ofNat (n * 4)
  let raw ← Metal.metalAlloc bytes
  pure { raw := raw, size := n * 4 }

/-- Free a typed buffer (returns to LRU). -/
def free (b : BufferHandle) : IO Unit :=
  Metal.metalFree b.raw (USize.ofNat b.size)

/-- Write a `FloatArray` into the buffer. -/
def writeF32 (b : BufferHandle) (vals : FloatArray) : IO Unit :=
  Metal.metalBufferWriteF32 b.raw vals

/-- Read a single f32 element. -/
def readF32At (b : BufferHandle) (i : Nat) : IO Float :=
  Metal.metalBufferReadF32 b.raw (USize.ofNat i)

/-- Write a `ByteArray` of host bytes into the buffer (offset 0). -/
def writeBytes (b : BufferHandle) (bytes : ByteArray) : IO Unit :=
  Metal.metalBufferWriteBytes b.raw bytes

/-- Read `n` bytes from the buffer into a fresh `ByteArray`. -/
def readBytes (b : BufferHandle) (n : Nat) : IO ByteArray :=
  Metal.metalBufferReadBytes b.raw (USize.ofNat n)

end Buffer

end Runtime

end Tgrad
