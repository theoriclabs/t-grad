-- Tgrad — top-level re-exports.
--
-- Module paths below mirror README §1's project shape exactly.
-- Commented entries are stubs for future gates; uncomment as the
-- corresponding lift lands.

-- L0 (scaffold, green)
import Tgrad.Dtype
import Tgrad.Shape
import Tgrad.UOp

-- L1 (types & graph-rewrite engine — green)
import Tgrad.UPat
import Tgrad.GraphRewrite
import Tgrad.Rules.Symbolic

-- L2 (schedule — green)
import Tgrad.Schedule.Indexing
import Tgrad.Schedule.Rangeify
import Tgrad.Schedule.Memory
import Tgrad.Schedule.Linear
import Tgrad.Schedule.Item

-- L3 (codegen + renderer — green)
import Tgrad.Codegen.Linearize
import Tgrad.Codegen.Simplify
import Tgrad.Codegen.GpuDims
import Tgrad.Codegen.Opt.Tc
import Tgrad.Codegen.Opt.IsTcEligible
import Tgrad.Codegen.Opt.Apply
import Tgrad.Codegen.Opt.Heuristic
import Tgrad.Renderer.Base
import Tgrad.Renderer.CStyle
import Tgrad.Renderer.CodeForOp
import Tgrad.Renderer.WmmaArgs
import Tgrad.Renderer.Metal
import Tgrad.Renderer.Elementwise
import Tgrad.Renderer.Creation
import Tgrad.Renderer.Cuda
import Tgrad.Renderer.MatmulScalar
import Tgrad.Renderer.MatmulTc

-- L4 (FFI runtime — green)
import Tgrad.Runtime.MetalDevice
import Tgrad.Runtime.MetalAllocator
import Tgrad.Runtime.MetalProgram
import Tgrad.Runtime.Buffer
import Tgrad.Runtime.Cache
import Tgrad.Runtime.Cuda

-- L5 (pipeline + Tensor — active)
import Tgrad.Tensor
import Tgrad.Pipeline

-- L6 (Python authoring layer — Python → C → Lean via @[export])
import Tgrad.PythonFFI

/-! # Tgrad — top-level re-exports

  Product modules only. The checked ontology, findings, live conditions,
  and work graph live under the separate `TgradSpec` library so they govern
  the product without becoming part of its runtime shared object.
-/
