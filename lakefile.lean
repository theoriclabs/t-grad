import Lake
open Lake DSL

package tgrad where
  -- v1 scope per `docs/drafts/3_tgrad_design/01_design.md`:
  -- bf16 Metal matmul + ultrathin Python authoring layer.

@[default_target]
lean_lib Tgrad where
  precompileModules := true

-- CLI entry: `tgrad <subcommand>` per design doc §9.
-- L4 (FFI runtime) links the Metal Obj-C bridge from c/.
-- The .o files are produced by `c/Makefile` and must exist before
-- `lake build tgrad-cli`.
def macosSdkPath : String :=
  "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

def metalLinkArgs : Array String := #[
  "c/build/metal_alloc.o",
  "c/build/metal_alloc_lean.o",
  "-isysroot", macosSdkPath,
  "-Wl,-syslibroot," ++ macosSdkPath,
  "-F" ++ macosSdkPath ++ "/System/Library/Frameworks",
  "-framework", "Metal",
  "-framework", "Foundation",
  "-framework", "CoreFoundation",
  "-framework", "CoreGraphics",
  "-lobjc"
]

lean_exe «tgrad-cli» where
  root := `Main
  supportInterpreter := true
  moreLinkArgs := metalLinkArgs

-- Layer-2 Lean test orchestrator. Replaces the 17 per-phase Mains
-- with one entry that walks each module's unit fixtures.
-- Same moreLinkArgs as tgrad-cli since Tests.lean transitively
-- imports the Runtime modules.
lean_exe «tgrad-tests» where
  root := `Tests
  supportInterpreter := true
  moreLinkArgs := metalLinkArgs
