#!/usr/bin/env python3
"""Regenerate the reviewed tinygrad parity-oracle classification.

The decisions below are intentionally pinned to one upstream revision.  Imports
are extracted from that revision at generation time so the JSON remains an
auditable product of the reviewed source instead of a second hand-maintained
copy of it.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Final


UPSTREAM_COMMIT: Final = "19c4d736f2bc8e26d21f08b28ffd6298408da00f"
GROUPS: Final = ("null", "unit", "backend")
CLASSES: Final = ("api_surface", "internal_repr", "infrastructure", "ambiguous")
CLASS_DEFINITIONS: Final = {
  "api_surface": "Asserts representation-independent, observable behavior of the public Tensor, dtype, or device API.",
  "internal_repr": "Asserts tinygrad-specific UOp, graph, codegen, renderer, scheduler, optimizer, or runtime implementation details.",
  "infrastructure": "Asserts tinygrad's supporting tools, services, interchange formats, caches, loaders, or visualization/profiling facilities.",
  "ambiguous": "The file does not contain enough evidence to assign one of the other classes without guessing.",
}

# Each entry was reviewed from the imports and representative test bodies at
# UPSTREAM_COMMIT.  This is policy data, not an inference from Tgrad behavior.
DECISIONS: Final = {
  "null": {
    "__init__.py": ("ambiguous", "The empty package initializer has no imports, test bodies, or assertions from which to infer a contract."),
    "test_attention.py": ("internal_repr", "It imports TinyJit and UOp and asserts schedule program counts and JIT-cache shape rather than attention values."),
    "test_autogen.py": ("infrastructure", "It imports tinygrad.runtime.support.autogen and validates generated ctypes bindings and C interop."),
    "test_compile_failures.py": ("internal_repr", "It imports tinygrad.codegen and compile_linear and checks compiler/code-generation failures and emitted instructions."),
    "test_const_folding.py": ("internal_repr", "It imports Ops and UOp and asserts rewritten node opcodes, dtypes, arguments, and constant graph structure."),
    "test_device.py": ("internal_repr", "It imports Compiler and runtime compiler implementations and asserts renderer/compiler selection and compilation-cache internals alongside device parsing."),
    "test_disk_cache.py": ("infrastructure", "It imports the diskcache helpers and tests cache tables, serialization, decorators, and cross-process behavior."),
    "test_dtype.py": ("api_surface", "It imports Tensor and public dtype objects and asserts dtype conversion, naming, singleton, pickling, and numeric cast results."),
    "test_dtype_spec.py": ("api_surface", "It imports Tensor, Device, and dtype functions and checks public dtype predicates, promotion, truncation, and values against NumPy and PyTorch."),
    "test_elf.py": ("infrastructure", "It imports the ELF loader and Clang compiler support and tests object loading, linking, and external-symbol diagnostics."),
    "test_gc.py": ("internal_repr", "It imports Buffer, run_linear, and UOp and asserts object counts, reference counts, and graph lifetime behavior."),
    "test_gpudims.py": ("internal_repr", "It imports codegen.gpudims, UOp, Ops, and Renderer and asserts SPECIAL/RANGE graph layout and grouped-dimension lowering."),
    "test_gradient.py": ("internal_repr", "It imports compute_gradient and UOp and evaluates symbolic gradient UOp expressions and realization state directly."),
    "test_graph_rewrite.py": ("internal_repr", "It imports graph-rewrite, pattern, and UOp primitives and asserts rewritten opcode, identity, source, and argument structure."),
    "test_hcq_iface.py": ("internal_repr", "It imports low-level HCQ, USB, and MMIO interfaces and tests their memory-view and register-access mechanics directly."),
    "test_helpers.py": ("infrastructure", "It imports tinygrad.helpers broadly and tests context variables, formatting, fetching, memory views, decorators, and utility functions."),
    "test_indexing.py": ("api_surface", "It imports only Tensor from tinygrad and asserts public indexing shapes and user-facing index errors."),
    "test_linearizer_failures.py": ("internal_repr", "It constructs UOp kernels and invokes to_program to test a specific linearizer failure case."),
    "test_linearizer_rewrite.py": ("internal_repr", "It imports to_program and optimizer options and asserts rewritten program sources, applied options, and kernel metadata."),
    "test_llm_server.py": ("infrastructure", "It imports the LLM server and validates its OpenAI-compatible streaming, request, response, and usage protocol."),
    "test_llm_tokenizer.py": ("infrastructure", "It imports the LLM tokenizer and template helpers and tests tokenization tables, templates, and tokenizer performance."),
    "test_memory_planner.py": ("internal_repr", "It imports schedule.memory and UOp and asserts arena reuse, aliasing, offsets, pinning, and replacement-map identity."),
    "test_method_cache.py": ("internal_repr", "It replaces Device.compiler.compile_cached and verifies that equivalent Tensor graphs reuse compiled methods."),
    "test_microbenchmarks.py": ("internal_repr", "It imports UOp and profiles UOp construction, simplification, and topological graph size."),
    "test_mnist_dataset.py": ("infrastructure", "It imports the MNIST dataset loader and checks loader realization through a global kernel counter."),
    "test_multitensor.py": ("internal_repr", "It imports JIT and Ops and primarily asserts multi-buffer allocation sizes, kernel counts, and scheduler memory behavior."),
    "test_pattern_matcher.py": ("internal_repr", "It imports PatternMatcher, UPat, Ops, and UOp and asserts pattern fields, matches, captures, and rewritten node identity."),
    "test_process_replay.py": ("infrastructure", "It imports the external process-replay utility and checks reproducible replay of codegen programs and optimization choices."),
    "test_real_world.py": ("internal_repr", "It imports TinyJit and GlobalCounters and gates model graphs on compiled-kernel counts, memory use, and JIT capture coverage rather than model outputs."),
    "test_rearrange_einops.py": ("api_surface", "It imports Tensor and asserts observable rearrange shapes and element ordering."),
    "test_resnet.py": ("infrastructure", "It imports extra.models.resnet and exercises pretrained-weight loading rather than Tensor numerical behavior."),
    "test_rewrite_bottom_up_gate.py": ("internal_repr", "It imports BottomUpGate, PatternMatcher, graph_rewrite, and UOp and checks traversal and rewrite ordering."),
    "test_schedule.py": ("internal_repr", "It imports scheduler, UOp, codegen, and realization primitives and extensively asserts schedule nodes, fusion, buffers, and kernel counts."),
    "test_schedule_cache.py": ("internal_repr", "It imports schedule_cache and asserts cache entries, variable bindings, and scheduler event counts."),
    "test_simplify_valid_idx.py": ("internal_repr", "It imports indexing and symbolic rewrite passes and asserts rendered predicates and rewritten UOp validity structure."),
    "test_symbolic_failures.py": ("internal_repr", "It imports Variable but directly tests UOp simplify, substitute, and ssimplify behavior on symbolic expression graphs."),
    "test_symbolic_tensor.py": ("internal_repr", "It imports the private _broadcast_shape helper and asserts UOp bindings, unbind maps, and contiguous-view offsets in addition to shapes."),
    "test_tensor.py": ("internal_repr", "It imports Ops, UOp, renderers, and to_program and asserts STORE/INDEX nodes, generated dtypes, and buffer identity alongside Tensor behavior."),
    "test_tensor_io.py": ("infrastructure", "It imports TensorIO and tests its file-like construction, seek, and bounds behavior."),
    "test_tensor_metadata.py": ("internal_repr", "It imports private metadata and capture state and asserts metadata attached to Tensor UOps and scheduled calls."),
    "test_tensor_uop_mixin.py": ("internal_repr", "It imports UOp rewrite primitives and asserts Tensor operations produce the identical interned UOp objects as direct UOp operations."),
    "test_tensor_uop_representation.py": ("internal_repr", "It imports UPat, Ops, and UOp and asserts Tensor parent/child UOp identity, mutation, and realization representation."),
    "test_tqdm.py": ("infrastructure", "It imports tinygrad's tqdm implementation and compares its formatting, timing, iteration, and performance with tqdm."),
    "test_transcendental_helpers.py": ("internal_repr", "It imports codegen transcendental decomposition helpers and evaluates their generated UOp expressions directly."),
    "test_uop_graph.py": ("internal_repr", "It imports UOp graph and rewrite primitives and asserts node dtype, opcode, source ordering, identity, and graph simplification."),
    "test_uop_repr.py": ("internal_repr", "It imports UOp and asserts exact repr strings, including constructor arguments and shared-node formatting."),
    "test_uop_resolve.py": ("internal_repr", "It imports UOp and resolve and tests truth, integer, and float coercion of constant and symbolic UOp expressions."),
    "test_uop_symbolic.py": ("internal_repr", "It imports UOp symbolic, spec, and validation passes and asserts symbolic node identity, rendering, bounds, and rewrites."),
    "test_uop_vmin_vmax.py": ("internal_repr", "It imports UOp and Ops and asserts internal vmin/vmax analysis on constructed expression nodes."),
    "test_uops.py": ("internal_repr", "It imports UOp specs, rewrite passes, and opcodes and asserts graph typing, ordering, repr round trips, and node structure."),
    "test_uops_stats.py": ("internal_repr", "It imports codegen, Estimates, UOp, and optimizer options and asserts operation, memory, and generated-program estimates."),
    "test_upat_compile.py": ("internal_repr", "It imports UPat compilation internals and checks generated matcher code and rewrite behavior."),
    "test_validate_oob.py": ("internal_repr", "It constructs buffer-index UOps and asserts the internal verifier accepts or rejects specific graph patterns."),
    "test_viz.py": ("infrastructure", "It imports tinygrad.viz and tests rewrite serialization, visualization rendering, profiling display, and CLI behavior."),
    "test_winograd.py": ("internal_repr", "It asserts Winograd schedule counts and internal operation-count ratios, with only an incidental public dtype check."),
  },
  "unit": {
    "__init__.py": ("ambiguous", "The empty package initializer has no imports, test bodies, or assertions from which to infer a contract."),
    "test_allreduce.py": ("internal_repr", "It imports Ops and inspects scheduled COPY pairs, sinks, and internal copy dtypes for all-reduce strategies."),
    "test_assign.py": ("internal_repr", "It imports Ops and UOp and repeatedly asserts backing-buffer identity, hand-built store/after graphs, and exact kernel counts alongside values."),
    "test_attention.py": ("api_surface", "It imports Tensor and LLM layers and compares observable attention, rotary embedding, state, and top-k values with references."),
    "test_call.py": ("internal_repr", "It constructs UOp function bodies for Tensor.call and asserts symbolic PARAM substitution and custom-call scheduling as well as results."),
    "test_callify.py": ("api_surface", "It imports Tensor and dtypes and checks only the values and dtypes observable after the public callify operation."),
    "test_conv.py": ("api_surface", "It imports Tensor and checks convolution and activation values against direct and NumPy references."),
    "test_cpu.py": ("internal_repr", "It imports CPU renderers and to_program and asserts whether a specific assembly instruction is emitted."),
    "test_disk_tensor.py": ("infrastructure", "It imports state loaders, safetensor helpers, and DiskDevice and validates on-disk layouts, model files, and serialization behavior."),
    "test_dtype_spec.py": ("api_surface", "It imports Tensor, Device, and dtypes and compares creation, casting, reduction, and dtype results with NumPy and PyTorch."),
    "test_dtype_weak.py": ("internal_repr", "It imports UOp specs and dtype_from_uop and asserts weak-dtype node typing, verifier failures, virtual storage, and lowering details."),
    "test_function.py": ("internal_repr", "It imports custom-function internals, UOp, Ops, KernelInfo, and ProgramInfo and asserts graph names, captures, precompilation, and custom kernels."),
    "test_getitem_ops.py": ("internal_repr", "It supplements numeric indexing checks with exact GlobalCounters operation budgets that constrain optimizer implementation."),
    "test_gguf.py": ("infrastructure", "It imports tinygrad.llm.gguf and generated GGML bindings and validates GGUF parsing and quantized interchange decoding."),
    "test_gradient.py": ("internal_repr", "It imports UOp, Ops, and KernelInfo and constructs implicit-broadcast and multi-output gradient graphs and custom kernels alongside public gradient checks."),
    "test_hashing.py": ("api_surface", "It uses Tensor hashing operations and compares output bytes with standard SHA-3 and SHAKE reference implementations."),
    "test_hcq_graph.py": ("internal_repr", "It imports HCQGraph and constructs PROGRAM UOps to assert graph-runner support decisions for runtime interfaces."),
    "test_helpers.py": ("infrastructure", "It imports helper utilities and tests polynomial evaluation, ndarray detection, and the garbage-collection decorator."),
    "test_indexing.py": ("api_surface", "It imports Tensor and compares public indexing and assignment results and errors with NumPy."),
    "test_invalid_tensor.py": ("internal_repr", "It imports Buffer, run_linear, Invalid, Ops, and UOp and directly executes and inspects invalid-sentinel UOp graphs."),
    "test_jit.py": ("internal_repr", "It imports JIT internals and UOp and asserts cache lengths, capture contents, beam activation, and compilation behavior beyond numeric results."),
    "test_jit_cases.py": ("api_surface", "It imports only TinyJit and Tensor and asserts observable results for explicit and implicit JIT inputs and outputs."),
    "test_jit_footguns.py": ("api_surface", "It exercises TinyJit through Tensor inputs and asserts user-visible aliasing results, errors, and workarounds."),
    "test_linalg.py": ("api_surface", "It imports Tensor and compares SVD, QR, and eigendecomposition results with algebraic and NumPy references."),
    "test_llm_mla.py": ("api_surface", "It imports Tensor and LLM model layers and asserts observable MLA output equivalence, state keys, and output shape."),
    "test_llm_moe.py": ("api_surface", "It imports Tensor and transformer blocks and compares mixture-of-experts outputs with numerical references."),
    "test_llm_server.py": ("infrastructure", "It tests model generation cache reuse, chunking, and sampling controls that support the LLM serving workflow."),
    "test_masked_tensor.py": ("api_surface", "It imports Tensor and asserts masked multiplication and addition shapes and values."),
    "test_metal_graph.py": ("internal_repr", "It constructs mock UOp buffers and asserts MetalGraph support decisions based on internal buffer opcodes and offsets."),
    "test_multitensor.py": ("internal_repr", "It imports Ops and UOp and asserts sharding-axis fields, schedule nodes, buffer layout, copy topology, and kernel counts alongside values."),
    "test_objc.py": ("infrastructure", "It imports the Objective-C runtime support layer and tests descriptor generation for class methods."),
    "test_randomness.py": ("api_surface", "It imports Tensor and nn and statistically compares public random distributions and initializer outputs with NumPy and PyTorch."),
    "test_realize_is_realize.py": ("internal_repr", "It imports Tensor but asserts the is_realized state of Tensor UOps and their sources directly."),
    "test_rearrange_einops.py": ("api_surface", "It imports Tensor and compares public rearrange shapes and values with NumPy/einops references."),
    "test_schedule_cache.py": ("internal_repr", "It imports schedule_cache and KernelInfo and asserts cache sizes and reuse for scheduled and custom kernels."),
    "test_setitem_schedule.py": ("internal_repr", "It asserts exact kernel and memory counters around setitem scheduling in addition to final Tensor values."),
    "test_shm_tensor.py": ("api_surface", "It uses Tensor and Device with SHM storage and asserts cross-process numerical round trips."),
    "test_symbolic_tensor.py": ("api_surface", "It imports Variable and Tensor and asserts the public padded Tensor values for a bound symbolic extent."),
    "test_system_pci_scan_bus.py": ("infrastructure", "It mocks sysfs and tests the runtime system's PCI device-discovery helper."),
    "test_tar.py": ("infrastructure", "It imports tar_extract and validates archive keys, contents, sizes, and error handling."),
    "test_tensor_data.py": ("api_surface", "It imports Tensor and dtypes and asserts public bytes, data memoryview, shape, format, and value behavior."),
    "test_tensor_io.py": ("infrastructure", "It imports TensorIO and tests its file-like read and end-of-file behavior."),
    "test_tinyfs.py": ("infrastructure", "It imports fs_store, fs_load, and tinyfs hashing helpers and tests storage, chunking, hashes, and round trips."),
    "test_winograd.py": ("api_surface", "It compares public convolution outputs with and without Winograd and with a padded convolution reference."),
  },
  "backend": {
    "__init__.py": ("ambiguous", "The empty package initializer has no imports, test bodies, or assertions from which to infer a contract."),
    "test_arange.py": ("internal_repr", "It imports compilation and estimate helpers and asserts schedule length and operation-count complexity in addition to arange values."),
    "test_asm_gemm.py": ("internal_repr", "It calls backend-specific extra assembly GEMM kernels directly, so its numerical checks validate a particular kernel implementation rather than Tensor matmul."),
    "test_call.py": ("internal_repr", "It builds CALL and custom-kernel UOps and tests their C-style renderer and foreign-function execution."),
    "test_const_folding.py": ("internal_repr", "It imports Ops and UOp and asserts deviceless CONST nodes, absent COPY schedule nodes, and folding-specific graph behavior."),
    "test_custom_kernel.py": ("internal_repr", "It constructs custom UOp kernels with KernelInfo, AxisType, and Ops and tests their execution and scheduling."),
    "test_dtype.py": ("api_surface", "It imports Tensor, Device, and dtypes and compares public cast, bitcast, promotion, overflow, and dtype values with NumPy and PyTorch."),
    "test_dtype_alu.py": ("api_surface", "It applies Tensor ALU operations across dtypes and compares observable scalar and array results with reference arithmetic."),
    "test_edgecases.py": ("api_surface", "It imports Tensor and nn and compares public edge-case values and errors with NumPy and PyTorch."),
    "test_encodings.py": ("internal_repr", "It imports x86 renderer registers and opcodes and asserts exact encoded instruction bytes."),
    "test_graph.py": ("internal_repr", "It imports MultiGraphRunner, Buffer, run_linear, and UOp and directly tests internal graph ordering and raw-buffer effects."),
    "test_interop.py": ("api_surface", "It transfers Tensor storage to and from PyTorch and compares observable values and writes."),
    "test_isel.py": ("internal_repr", "It imports x86 instruction selection internals and asserts selected opcodes and UOp source structure."),
    "test_jit.py": ("internal_repr", "It imports graph classes and UOp and asserts captured linear programs, graph calls, cache layout, and backend JIT internals."),
    "test_linearizer.py": ("internal_repr", "It imports codegen, UOp, Ops, renderers, and optimizer options and asserts linear node order, opcodes, ranges, and buffers."),
    "test_linearizer_dumb.py": ("internal_repr", "It constructs a UOp kernel with explicit optimizer options to reproduce a linearizer/beam failure."),
    "test_llama_kernels.py": ("internal_repr", "It calls backend-specific extra Llama and fused-attention kernels directly rather than testing the public Tensor operations they replace."),
    "test_multitensor.py": ("internal_repr", "It imports realization and UOp primitives and asserts exact COPY schedule positions, schedule counts, capture state, and kernel counts."),
    "test_nn.py": ("internal_repr", "It imports Ops and run_linear and asserts scheduled SINK counts, operation budgets, and internal sharding-axis fields alongside layer values."),
    "test_ops.py": ("api_surface", "It exercises the public Tensor operation surface and compares shapes, dtypes, values, and exceptions primarily with PyTorch and NumPy."),
    "test_opt_gemm.py": ("internal_repr", "It injects explicit codegen optimizer options into UOp GEMM programs and validates those specialized generated kernels."),
    "test_optim.py": ("api_surface", "It imports public optimizers and compares parameter updates, losses, devices, and errors with PyTorch."),
    "test_pickle.py": ("infrastructure", "It tests serialization of code objects, pattern matchers, UOps, buffers, Tensor state, JIT objects, and context variables."),
    "test_profiler.py": ("infrastructure", "It imports profiling event types and validates kernel, copy, graph, timing, and device-profile event streams."),
    "test_quantize_onnx.py": ("infrastructure", "It imports ONNX tooling and the OnnxRunner and validates model quantization, loading, and interchange execution."),
    "test_randomness.py": ("internal_repr", "It imports codegen and renderer internals and asserts generated opcodes, private RNG caches, tensor counts, and kernel reuse as well as distributions."),
    "test_rangeify.py": ("internal_repr", "It imports the rangeify rewrite pass, UPat, Ops, UOp, and optimizer options and tests that lowering transformation directly."),
    "test_renderer_failures.py": ("internal_repr", "It constructs UOp programs for several renderers and asserts generated source shape and renderer-specific execution failures."),
    "test_schedule.py": ("internal_repr", "It imports UOp scheduling and realization primitives and asserts fusion, COPY nodes, realization state, and internal schedule structure."),
    "test_setitem.py": ("api_surface", "It imports Tensor and compares public setitem values, broadcasting, errors, aliasing results, and JIT behavior with NumPy."),
    "test_softmax_fusion.py": ("internal_repr", "It targets fusion by asserting schedule program counts and fused-kernel behavior in addition to numerical equivalence."),
    "test_stunning.py": ("api_surface", "It uses Tensor, Variable, and nn and asserts observable symbolic indexing, binding errors, and training values."),
    "test_subbuffer.py": ("internal_repr", "It imports Buffer directly and tests sub-buffer allocation, offsets, views, transfer, deallocation, and aliasing mechanics."),
    "test_symbolic_jit.py": ("api_surface", "It compares public symbolic TinyJit results and errors with concrete Tensor executions."),
    "test_symbolic_ops.py": ("api_surface", "It compares public Tensor operations at symbolic and concrete shapes by their numerical outputs."),
    "test_tensor.py": ("internal_repr", "It imports UOp and asserts deviceless UOp identity, internal device state, backing-buffer identity, and buffer markers alongside public Tensor results."),
    "test_tensor_variable.py": ("internal_repr", "It uses Tensor and Variable but directly asserts a Tensor UOp's backing-buffer size and symbolic UOp shape alongside public values."),
    "test_to_numpy.py": ("api_surface", "It imports Tensor and asserts that numpy returns an ndarray with the expected shape and values."),
    "test_transcendental.py": ("api_surface", "It compares public Tensor transcendental results and special values with NumPy across supported dtypes."),
    "test_uops.py": ("internal_repr", "It builds and executes UOp programs and asserts generated opcode presence, renderer behavior, and low-level kernel results."),
    "test_wait_loop.py": ("internal_repr", "It constructs UOp loop, wait, buffer, and kernel graphs and tests their low-level runtime execution."),
    "test_zero_copy.py": ("internal_repr", "It uses a bandwidth threshold to require a particular zero-copy runtime implementation rather than asserting Tensor semantics."),
  },
}


def parse_args() -> argparse.Namespace:
  repo_root = Path(__file__).resolve().parents[2]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--tinygrad-root", type=Path, default=Path("/tmp/tg_oracle/tinygrad"),
                      help="path to the pinned tinygrad checkout")
  parser.add_argument("--output", type=Path, default=repo_root / "fixtures/parity/oracle_classification.json",
                      help="JSON path to write")
  parser.add_argument("--check", action="store_true", help="fail if the output is not already up to date")
  return parser.parse_args()


def checkout_commit(root: Path) -> str:
  try:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
  except (OSError, subprocess.CalledProcessError) as exc:
    raise SystemExit(f"cannot read tinygrad revision from {root}: {exc}") from exc


def source_imports(path: Path) -> list[str]:
  tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
  nodes = sorted((node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))),
                 key=lambda node: (node.lineno, node.col_offset))
  return [ast.unparse(node) for node in nodes]


def validate_inventory(tinygrad_root: Path) -> None:
  errors: list[str] = []
  for group in GROUPS:
    actual = {path.name for path in (tinygrad_root / "test" / group).glob("*.py")}
    reviewed = set(DECISIONS[group])
    if missing := sorted(actual - reviewed): errors.append(f"{group}: unclassified files: {', '.join(missing)}")
    if stale := sorted(reviewed - actual): errors.append(f"{group}: decisions without files: {', '.join(stale)}")
  for group, decisions in DECISIONS.items():
    if group not in GROUPS: errors.append(f"unknown group in decisions: {group}")
    for filename, (classification, rationale) in decisions.items():
      if classification not in CLASSES: errors.append(f"{group}/{filename}: unknown class {classification}")
      if not rationale.endswith("."): errors.append(f"{group}/{filename}: rationale is not a sentence")
  if errors: raise SystemExit("classification inventory mismatch:\n  " + "\n  ".join(errors))


def build_document(tinygrad_root: Path) -> dict:
  commit = checkout_commit(tinygrad_root)
  if commit != UPSTREAM_COMMIT:
    raise SystemExit(f"expected tinygrad {UPSTREAM_COMMIT}, found {commit} at {tinygrad_root}")
  validate_inventory(tinygrad_root)

  files = []
  by_class: Counter[str] = Counter()
  by_group: dict[str, dict] = {}
  for group in GROUPS:
    group_counts: Counter[str] = Counter()
    for filename in sorted(DECISIONS[group]):
      classification, rationale = DECISIONS[group][filename]
      files.append({
        "group": group,
        "filename": filename,
        "class": classification,
        "rationale": rationale,
        "imports": source_imports(tinygrad_root / "test" / group / filename),
      })
      by_class[classification] += 1
      group_counts[classification] += 1
    by_group[group] = {
      "files": sum(group_counts.values()),
      "by_class": {name: group_counts[name] for name in CLASSES},
    }

  return {
    "schema_version": 1,
    "upstream": {
      "repository": "tinygrad/tinygrad",
      "commit": commit,
      "test_directories": [f"test/{group}" for group in GROUPS],
    },
    "classification_policy": {
      "unit": "file",
      "basis": "Imports and representative asserted behavior in the pinned upstream source; no Tgrad implementation or score was consulted.",
      "mixed_file_rule": "A file with substantive tinygrad-specific representation, codegen, renderer, scheduler, optimizer, or runtime assertions is internal_repr even when it also contains public numerical checks.",
      "classes": CLASS_DEFINITIONS,
    },
    "totals": {
      "files": len(files),
      "by_class": {name: by_class[name] for name in CLASSES},
      "by_group": by_group,
    },
    "files": files,
  }


def main() -> int:
  args = parse_args()
  rendered = json.dumps(build_document(args.tinygrad_root.resolve()), indent=2, sort_keys=False) + "\n"
  if args.check:
    try: current = args.output.read_text(encoding="utf-8")
    except FileNotFoundError: current = ""
    if current != rendered:
      print(f"out of date: {args.output}", file=sys.stderr)
      return 1
    print(f"up to date: {args.output}")
    return 0
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(rendered, encoding="utf-8")
  print(f"wrote {args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
