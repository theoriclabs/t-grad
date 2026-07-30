import Tgrad.Backend.FillPlan

/-! # HIP/ROCm profile leaf

This module owns only AMD/HIP profile spelling and admission.  Tensor values,
dtype admission, byte arithmetic, launch geometry, and semantic identity stay
in `Tgrad.Backend.FillPlan`.
-/
namespace Tgrad.Backend.Hip

open Tgrad
open Tgrad.Backend

def backendIdentity : BackendIdentity :=
  (BackendIdentity.build "hip-rocm").toOption.get (by native_decide)

inductive CompilerTool where
  | hipcc
  | hiprtc
  deriving BEq, Repr, DecidableEq

def CompilerTool.stableName : CompilerTool → String
  | .hipcc => "hipcc"
  | .hiprtc => "hiprtc"

def CompilerTool.identity : CompilerTool → CompilerToolIdentity
  | .hipcc => (CompilerToolIdentity.build "hipcc").toOption.get (by native_decide)
  | .hiprtc => (CompilerToolIdentity.build "hiprtc").toOption.get (by native_decide)

def CompilerTool.parse (name : String) : Option CompilerTool :=
  match name with
  | "hipcc" => some .hipcc
  | "hiprtc" => some .hiprtc
  | _ => none

def CompilerTool.admitsMode : CompilerTool → CompilerMode → Bool
  | .hipcc, .offline => true
  | .hiprtc, .runtime => true
  | _, _ => false

def maxThreadsPerBlock : Nat := 1024

/-- AMD GPU architecture spelling admitted by this leaf.  The neutral spine
keeps architecture opaque and deliberately contains no `gfx` policy. -/
def validArchitecture (name : String) : Bool :=
  if !name.startsWith "gfx" then false
  else
    let suffix := (name.drop 3).toString
    match suffix.toList with
    | [] => false
    | first :: rest =>
        (suffix.length == 3 || suffix.length == 4) && first.isDigit &&
          rest.all (fun char =>
          char.isDigit || char == 'a' || char == 'b' || char == 'c' ||
            char == 'd' || char == 'e' || char == 'f')

inductive ProfileError where
  | identity (reason : IdentityError)
  | invalidArchitecture (name : String)
  | unsupportedCompilerTool (name : String)
  | compilerModeMismatch (mode : CompilerMode) (tool : CompilerTool)
  | invalidThreadLimit (limit : Nat)
  | wrongBackend (actual : String)
  deriving BEq, Repr

def buildProfile (architecture : String) (mode : CompilerMode)
    (toolName compilerVersion : String) (threadLimit : Nat) :
    Except ProfileError DeviceProfile := do
  if !validArchitecture architecture then
    throw (.invalidArchitecture architecture)
  let tool ← match CompilerTool.parse toolName with
    | some tool => pure tool
    | none => throw (.unsupportedCompilerTool toolName)
  if !tool.admitsMode mode then throw (.compilerModeMismatch mode tool)
  if threadLimit == 0 || threadLimit > maxThreadsPerBlock then
    throw (.invalidThreadLimit threadLimit)
  let architectureIdentity ← match ArchitectureIdentity.build architecture with
    | .ok value => pure value
    | .error reason => throw (.identity reason)
  let versionIdentity ← match CompilerVersionIdentity.build compilerVersion with
    | .ok value => pure value
    | .error reason => throw (.identity reason)
  match DeviceProfile.build backendIdentity architectureIdentity {
      mode, tool := tool.identity, version := versionIdentity } threadLimit with
  | .ok profile => pure profile
  | .error reason => throw (.identity reason)

/-- Revalidate opaque shared identities at the HIP leaf boundary.  This is
used both by rendering and by the sealed probe authority. -/
def validateProfile (profile : DeviceProfile) : Except ProfileError Unit := do
  if profile.backend != backendIdentity then
    throw (.wrongBackend profile.backend.stableName)
  if !validArchitecture profile.architecture.stableName then
    throw (.invalidArchitecture profile.architecture.stableName)
  let tool ← match CompilerTool.parse profile.compiler.tool.stableName with
    | some tool => pure tool
    | none => throw (.unsupportedCompilerTool profile.compiler.tool.stableName)
  if !tool.admitsMode profile.compiler.mode then
    throw (.compilerModeMismatch profile.compiler.mode tool)
  if profile.maxThreadsPerBlock.value == 0 ||
      profile.maxThreadsPerBlock.value > maxThreadsPerBlock then
    throw (.invalidThreadLimit profile.maxThreadsPerBlock.value)

def admitsProfile (profile : DeviceProfile) : Bool :=
  (validateProfile profile).isOk

inductive PlanBuildError where
  | profile (reason : ProfileError)
  | plan (reason : PlanError)
  deriving BEq, Repr

/-- HIP entry point for the shared planner.  It adds only vendor profile
admission before delegating all computation semantics to `FillPlan.build`. -/
def buildFillPlan (profile : DeviceProfile) (dtype : Dtype)
    (input : ScalarInput) (elementCount threadsPerBlock : Nat) :
    Except PlanBuildError FillPlan := do
  match validateProfile profile with
  | .ok () => pure ()
  | .error reason => throw (.profile reason)
  match FillPlan.build profile dtype input elementCount threadsPerBlock with
  | .ok plan => pure plan
  | .error reason => throw (.plan reason)

end Tgrad.Backend.Hip
