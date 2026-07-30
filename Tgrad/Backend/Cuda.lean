import Tgrad.Backend.FillPlan

/-! # CUDA-specific profile admission

The shared spine owns profile identity and all fill semantics. This leaf owns
only CUDA's architecture spelling and compiler tool/mode pairing.
-/
namespace Tgrad.Backend.Cuda

inductive CompilerTool where
  | nvrtc
  | nvcc
  deriving BEq, Repr, DecidableEq

def CompilerTool.stableName : CompilerTool → String
  | .nvrtc => "nvrtc"
  | .nvcc => "nvcc"

def CompilerTool.mode : CompilerTool → CompilerMode
  | .nvrtc => .runtime
  | .nvcc => .offline

inductive ProfileError where
  | invalidArchitecture (architecture : String)
  | invalidThreadLimit (limit : Nat)
  | invalidIdentity (reason : IdentityError)
  deriving BEq, Repr

def identity : BackendIdentity :=
  (BackendIdentity.build "cuda").toOption.get (by native_decide)

private def asciiDigit (char : Char) : Bool :=
  48 <= char.toNat && char.toNat <= 57

/-- Conservative CUDA binary architecture spelling admitted by this leaf.
Toolchain-specific support for a particular compute capability still belongs
to a real compiler/device probe. -/
def architectureNameAdmitted (architecture : String) : Bool :=
  let suffix := architecture.toList.drop 3
  architecture.startsWith "sm_" && suffix.length >= 2 && suffix.all asciiDigit

private def mapIdentity : Except IdentityError α → Except ProfileError α
  | .ok value => .ok value
  | .error reason => .error (.invalidIdentity reason)

def buildProfile (architecture : String) (tool : CompilerTool)
    (version : String) (maxThreadsPerBlock : Nat) :
    Except ProfileError DeviceProfile := do
  if !architectureNameAdmitted architecture then
    throw (.invalidArchitecture architecture)
  if maxThreadsPerBlock == 0 || maxThreadsPerBlock > 1024 then
    throw (.invalidThreadLimit maxThreadsPerBlock)
  let architectureIdentity ← mapIdentity (ArchitectureIdentity.build architecture)
  let toolIdentity ← mapIdentity (CompilerToolIdentity.build tool.stableName)
  let versionIdentity ← mapIdentity (CompilerVersionIdentity.build version)
  mapIdentity (DeviceProfile.build identity architectureIdentity {
    mode := tool.mode
    tool := toolIdentity
    version := versionIdentity
  } maxThreadsPerBlock)

/-- Exact vendor compatibility predicate used by the sealed probe authority.
It does not choose or repair any profile field. -/
def profileAdmitted (profile : DeviceProfile) : Bool :=
  profile.backend == identity &&
  architectureNameAdmitted profile.architecture.stableName &&
  profile.maxThreadsPerBlock.value <= 1024 &&
  ((profile.compiler.mode == .runtime &&
      profile.compiler.tool.stableName == CompilerTool.nvrtc.stableName) ||
    (profile.compiler.mode == .offline &&
      profile.compiler.tool.stableName == CompilerTool.nvcc.stableName))

end Tgrad.Backend.Cuda
