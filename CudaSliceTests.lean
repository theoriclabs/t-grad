import Tgrad.Renderer.Cuda
import Tgrad.Runtime.Cuda

open Tgrad

def contains (haystack needle : String) : Bool :=
  (haystack.splitOn needle).length > 1

def isError : Except ε α → Bool
  | .error _ => true
  | .ok _ => false

def isOk : Except ε α → Bool
  | .ok _ => true
  | .error _ => false

def check (name : String) (condition : Bool) : IO Nat := do
  if condition then
    IO.println ("PASS " ++ name)
    pure 0
  else
    IO.eprintln ("FAIL " ++ name)
    pure 1

def cudaProfile (compiler : Backend.CompilerMode := .nvrtc "12.4") :
    Backend.Profile := {
  backend := .cuda
  architecture := "sm_80"
  compiler
  maxThreadsPerBlock := 1024
}

def hipProfile : Backend.Profile := {
  backend := .hip
  architecture := "gfx1100"
  compiler := .hiprtc "6.1"
  maxThreadsPerBlock := 1024
}

def getPlan (profile : Backend.Profile) (dtype : Dtype)
    (value : Backend.RawFillValue) (count threads : Nat) : IO Backend.FillPlan :=
  match Backend.FillPlan.build profile dtype value count threads with
  | .ok plan => pure plan
  | .error error => throw (IO.userError ("focused test plan construction failed: " ++ reprStr error))

def hasLaunch (count threads blocks : Nat) : Bool :=
  match Backend.FillPlan.build cudaProfile .float32_ (.float32Bits 1065353216)
      count threads with
  | .ok plan => plan.blocksPerGrid.value == blocks
  | .error _ => false

def main : IO Unit := do
  let floatPlan ← getPlan cudaProfile .float32_ (.float32Bits 1065353216) 513 256
  let intPlan ← getPlan cudaProfile .int32_ (.signed (-7)) 17 16
  let hipPlan ← getPlan hipProfile .float32_ (.float32Bits 1065353216) 513 256
  let nvccPlan ← getPlan (cudaProfile (.nvcc "12.4")) .float32_
    (.float32Bits 1065353216) 513 256
  let .ok floatCuda := Renderer.Cuda.renderFill floatPlan
    | throw (IO.userError "CUDA f32 render unexpectedly rejected")
  let .ok intCuda := Renderer.Cuda.renderFill intPlan
    | throw (IO.userError "CUDA i32 render unexpectedly rejected")

  let mut failures := 0
  failures := failures + (← check "output_index"
    (contains floatCuda.source
       "(((unsigned long long)blockIdx.x * (unsigned long long)blockDim.x) + (unsigned long long)threadIdx.x)" &&
     contains floatCuda.source "out[idx]"))
  failures := failures + (← check "dtype_value_float32"
    (contains floatCuda.source "float* out" &&
     contains floatCuda.source "__uint_as_float(1065353216U)"))
  failures := failures + (← check "dtype_value_int32"
    (contains intCuda.source "int* out" &&
     contains intCuda.source "((int)-7)"))
  failures := failures + (← check "ceil_div_launch"
    (floatPlan.threadsPerBlock.value == 256 &&
     floatPlan.blocksPerGrid.value == 3 &&
     floatPlan.byteCount.value == 2052))
  failures := failures + (← check "ceil_div_boundaries"
    (hasLaunch 0 256 0 && hasLaunch 1 256 1 && hasLaunch 255 256 1 &&
     hasLaunch 256 256 1 && hasLaunch 257 256 2 && hasLaunch 512 256 2))
  failures := failures + (← check "maximum_grid_width_accepted"
    (hasLaunch (Backend.maxAbiU32 * 1024) 1024 Backend.maxAbiU32))
  failures := failures + (← check "source_exact_count_guard"
    (contains floatCuda.source "if (idx < 513ULL)"))
  failures := failures + (← check "cache_backend_identity"
    (floatPlan.cacheIdentity != hipPlan.cacheIdentity))
  failures := failures + (← check "cache_compiler_identity"
    (floatPlan.cacheIdentity != nvccPlan.cacheIdentity))
  let otherArchPlan ← getPlan
    { cudaProfile with architecture := "sm_90" }
    .float32_ (.float32Bits 1065353216) 513 256
  let otherValuePlan ← getPlan cudaProfile .float32_ (.float32Bits 2147483648) 513 256
  let otherLaunchPlan ← getPlan cudaProfile .float32_ (.float32Bits 1065353216) 513 128
  failures := failures + (← check "cache_architecture_identity"
    (floatPlan.cacheIdentity != otherArchPlan.cacheIdentity))
  failures := failures + (← check "cache_value_identity"
    (floatPlan.cacheIdentity != otherValuePlan.cacheIdentity))
  failures := failures + (← check "cache_launch_identity"
    (floatPlan.cacheIdentity != otherLaunchPlan.cacheIdentity))
  failures := failures + (← check "kernel_identity_is_complete"
    (floatPlan.kernelName != hipPlan.kernelName &&
     floatPlan.kernelName != nvccPlan.kernelName &&
     floatPlan.kernelName != otherArchPlan.kernelName &&
     floatPlan.kernelName != otherValuePlan.kernelName &&
     floatPlan.kernelName != otherLaunchPlan.kernelName))
  failures := failures + (← check "renderer_cache_uses_plan_identity"
    (contains floatCuda.cacheIdentity floatPlan.cacheIdentity))
  failures := failures + (← check "unsupported_dtype_rejected"
    (isError (Backend.FillPlan.build cudaProfile .float64_ (.float32Bits 0) 1 1)))
  failures := failures + (← check "dtype_value_mismatch_rejected"
    (isError (Backend.FillPlan.build cudaProfile .int32_ (.float32Bits 0) 1 1)))
  failures := failures + (← check "zero_threads_rejected"
    (isError (Backend.FillPlan.build cudaProfile .float32_ (.float32Bits 0) 1 0)))
  failures := failures + (← check "byte_overflow_rejected"
    (isError (Backend.FillPlan.build
      { cudaProfile with maxThreadsPerBlock := Backend.maxAbiU32 }
      .float32_ (.float32Bits 0)
      (Backend.maxAbiU32 * Backend.maxAbiU32) Backend.maxAbiU32)))
  failures := failures + (← check "element_count_abi_overflow_rejected"
    (isError (Backend.FillPlan.build cudaProfile .float32_ (.float32Bits 0)
      (Backend.maxAbiU64 + 1) 1)))
  failures := failures + (← check "launch_width_overflow_rejected"
    (isError (Backend.FillPlan.build cudaProfile .float32_ (.float32Bits 0)
      (Backend.maxAbiU32 * 1024 + 1) 1024)))
  failures := failures + (← check "profile_block_limit_rejected"
    (isError (Backend.FillPlan.build cudaProfile .float32_ (.float32Bits 0)
      1 1025)))
  failures := failures + (← check "wrong_backend_renderer_rejected"
    (isError (Renderer.Cuda.renderFill hipPlan)))
  let forgedInvalidProfile : Backend.FillPlan :=
    { floatPlan with profile := { floatPlan.profile with architecture := "" } }
  failures := failures + (← check "forged_invalid_profile_renderer_rejected"
    (isError (Renderer.Cuda.renderFill forgedInvalidProfile)))
  failures := failures + (← check "absent_probe_is_false"
    (!(Runtime.Cuda.availabilityFromProbe none).isAvailable))
  failures := failures + (← check "negative_probe_is_false"
    (!(Runtime.Cuda.availabilityFromProbe
      (some Runtime.Cuda.ProbeResult.runtimeMissing)).isAvailable))
  let localState ← Runtime.Cuda.localAvailability
  failures := failures + (← check "local_boundary_is_false"
    (!localState.isAvailable && isError (Runtime.Cuda.requireAvailable localState)))
  failures := failures + (← check "compile_request_exact_source_accepted"
    (isOk (Runtime.Cuda.CompileRequest.build floatPlan floatCuda)))
  let forgedSource : Renderer.Cuda.KernelSource :=
    { floatCuda with source := floatCuda.source ++ "// changed" }
  failures := failures + (← check "compile_request_forged_source_rejected"
    (isError (Runtime.Cuda.CompileRequest.build floatPlan forgedSource)))
  let validBuffer : Runtime.Cuda.DeviceBuffer := {
    backend := .cuda
    deviceOrdinal := 0
    ownership := .runtimeOwned
    byteCount := floatPlan.byteCount
    handle := 1
  }
  failures := failures + (← check "typed_copy_request_accepted"
    (isOk (Runtime.Cuda.CopyRequest.build .hostToDevice floatPlan.byteCount validBuffer)))
  let tooManyBytes : Backend.AbiU64 := { value := 2053, fits := by decide }
  failures := failures + (← check "copy_size_rejected"
    (isError (Runtime.Cuda.CopyRequest.build .hostToDevice tooManyBytes validBuffer)))
  let wrongBackendBuffer : Runtime.Cuda.DeviceBuffer :=
    { validBuffer with backend := .hip }
  failures := failures + (← check "copy_backend_rejected"
    (isError (Runtime.Cuda.CopyRequest.build .hostToDevice
      floatPlan.byteCount wrongBackendBuffer)))
  let nullBuffer : Runtime.Cuda.DeviceBuffer := { validBuffer with handle := 0 }
  failures := failures + (← check "copy_null_handle_rejected"
    (isError (Runtime.Cuda.CopyRequest.build .deviceToHost
      floatPlan.byteCount nullBuffer)))
  let unavailableModule : Runtime.Cuda.CompiledModule := {
    backend := .cuda
    deviceOrdinal := 0
    cacheIdentity := floatCuda.cacheIdentity
    handle := 1
  }
  failures := failures + (← check "launch_unavailable_rejected"
    (isError (Runtime.Cuda.LaunchRequest.build localState floatPlan
      unavailableModule validBuffer)))
  failures := failures + (← check "synchronize_unavailable_rejected"
    (isError (Runtime.Cuda.SynchronizeRequest.build localState .device)))

  if failures != 0 then
    IO.eprintln ("cuda-slice-tests: " ++ toString failures ++ " failure(s)")
    IO.Process.exit 1
  IO.println "cuda-slice-tests: all focused checks green"
