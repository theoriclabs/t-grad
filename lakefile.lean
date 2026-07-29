import Lake
open Lake DSL

package tgrad where
  -- v1 scope per `docs/drafts/3_tgrad_design/01_design.md`:
  -- bf16 Metal matmul + ultrathin Python authoring layer.

@[default_target]
lean_lib Tgrad where
  precompileModules := true

/-- Checked software and work specification. This is a separate root from
`Tgrad`, so roadmap/findings data is compiled by its own build/query targets
but is not linked into `Tgrad:shared`. -/
lean_lib TgradSpec where
  roots := #[`TgradSpec]
  precompileModules := true

/-- Human/agent query surface for the checked specification. -/
lean_exe «tgrad-spec» where
  root := `Tgrad.Spec.Work

-- CLI entry: `tgrad <subcommand>` per design doc §9.
-- L4 (FFI runtime) links the Metal Obj-C bridge from c/.
-- The .o files are produced by `c/Makefile` and must exist before
-- `lake build tgrad-cli`.
-- SDK resolution must match `c/Makefile`'s `SDK_PATH`, which asks
-- `xcrun`. Hardcoding a path breaks on Command-Line-Tools-only hosts
-- and on any host whose SDK is versioned (`MacOSX15.5.sdk`), which is
-- the default — the unversioned `MacOSX.sdk` symlink is not always
-- present. Order: $TGRAD_MACOS_SDK, then `xcrun`, then the legacy path.
def macosSdkFallback : String :=
  "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk"

unsafe def macosSdkPathImpl : String :=
  match unsafeIO (do
    if let some p ← IO.getEnv "TGRAD_MACOS_SDK" then
      if !p.trim.isEmpty then return p.trim
    let out ← IO.Process.output
      { cmd := "xcrun", args := #["--sdk", "macosx", "--show-sdk-path"] }
    return if out.exitCode == 0 && !out.stdout.trim.isEmpty
           then out.stdout.trim else macosSdkFallback) with
  | .ok p    => p
  | .error _ => macosSdkFallback

@[implemented_by macosSdkPathImpl]
def macosSdkPath : String := macosSdkFallback

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

/-- CPU-only focused checks for the backend-neutral accelerator spine. -/
lean_exe «backend-shared-tests» where
  root := `BackendSharedTests
