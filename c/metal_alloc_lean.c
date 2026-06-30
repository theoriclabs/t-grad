/*
 * metal_alloc_lean.c — Lean-runtime-aware wrappers for metal_alloc.m.
 *
 * Lean's @[extern] IO-returning functions follow a specific calling
 * convention: take their normal args plus a trailing `lean_object* w`
 * (the IO world token), and return `lean_obj_res` wrapping the boxed
 * value. Concretely:
 *
 *   Lean: @[extern "X"] opaque X (a : USize) : IO UInt64
 *   C:    lean_obj_res X(size_t a, lean_object* w);
 *
 * Wrapping in IO (rather than letting Lean type the call as a pure
 * USize→UInt64 function) prevents the compiler from CSE-ing two
 * `alloc 1024` calls into one — which would defeat the LRU-cache
 * round-trip test that is G1's gate.
 *
 * The underlying portable ObjC functions live in metal_alloc.m and
 * stay free of any lean.h dependency.
 */
#include <lean/lean.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <alloca.h>

/* Forward decls for the portable layer (metal_alloc.m). */
extern int    theograd_metal_available(void);
extern void*  theograd_metal_alloc(size_t size);
extern void   theograd_metal_free(void* ptr, size_t size);
extern int    theograd_metal_lru_count(void);
extern void   theograd_metal_lru_flush(void);
extern size_t theograd_metal_buffer_length(void* ptr);
extern void*  theograd_metal_compile(const char* msl_source);
extern void   theograd_metal_library_release(void* lib_ptr);
extern int    theograd_metal_library_function_count(void* lib_ptr);
extern void*  theograd_metal_buffer_contents(void* buf_ptr);
extern int    theograd_metal_dispatch(
                  void* library_ptr, const char* fn_name,
                  void* const* buffers, size_t n_buffers,
                  size_t gx, size_t gy, size_t gz,
                  size_t lx, size_t ly, size_t lz);

/* IO Bool — returns 1 if a default Metal device is available. */
lean_obj_res lean_theograd_metal_available(lean_object* w) {
    (void)w;
    return lean_io_result_mk_ok(lean_box(theograd_metal_available() ? 1 : 0));
}

/* IO UInt64 — alloc-or-cache-hit. Returns 0 on failure. */
lean_obj_res lean_theograd_metal_alloc(size_t size, lean_object* w) {
    (void)w;
    void* p = theograd_metal_alloc(size);
    return lean_io_result_mk_ok(lean_box_uint64((uint64_t)(uintptr_t)p));
}

/* IO Unit — return-to-cache or release. */
lean_obj_res lean_theograd_metal_free(uint64_t ptr, size_t size, lean_object* w) {
    (void)w;
    theograd_metal_free((void*)(uintptr_t)ptr, size);
    return lean_io_result_mk_ok(lean_box(0));
}

/* IO USize — buffer length, for sanity-checking alloc'd pointers. */
lean_obj_res lean_theograd_metal_buffer_length(uint64_t ptr, lean_object* w) {
    (void)w;
    size_t len = theograd_metal_buffer_length((void*)(uintptr_t)ptr);
    return lean_io_result_mk_ok(lean_box_usize(len));
}

/* IO UInt32 — LRU live-count, for assertions. */
lean_obj_res lean_theograd_metal_lru_count(lean_object* w) {
    (void)w;
    return lean_io_result_mk_ok(lean_box_uint32((uint32_t)theograd_metal_lru_count()));
}

/* IO Unit — flush LRU (release every cached buffer). */
lean_obj_res lean_theograd_metal_lru_flush(lean_object* w) {
    (void)w;
    theograd_metal_lru_flush();
    return lean_io_result_mk_ok(lean_box(0));
}

/* IO UInt64 — compile MSL. Returns 0 on failure (no device / compile
 * error). Lean's String passes as `lean_object*` with the UTF-8 bytes
 * accessible via `lean_string_cstr`. */
lean_obj_res lean_theograd_metal_compile(b_lean_obj_arg src, lean_object* w) {
    (void)w;
    const char* s = lean_string_cstr(src);
    void* lib = theograd_metal_compile(s);
    return lean_io_result_mk_ok(lean_box_uint64((uint64_t)(uintptr_t)lib));
}

/* IO Unit — release a library returned by metalCompile. */
lean_obj_res lean_theograd_metal_library_release(uint64_t lib_ptr, lean_object* w) {
    (void)w;
    theograd_metal_library_release((void*)(uintptr_t)lib_ptr);
    return lean_io_result_mk_ok(lean_box(0));
}

/* IO UInt32 — function count of a compiled library (-1 = NULL lib). */
lean_obj_res lean_theograd_metal_library_function_count(uint64_t lib_ptr, lean_object* w) {
    (void)w;
    int n = theograd_metal_library_function_count((void*)(uintptr_t)lib_ptr);
    /* Wrap negative as a sentinel UInt32; Lean test checks for the
     * positive count. */
    return lean_io_result_mk_ok(lean_box_uint32((uint32_t)n));
}

/* ======================================================================
 * G2 (b): host I/O on buffers + dispatch.
 *
 * Buffer I/O uses MTLResourceStorageModeShared (set in
 * theograd_metal_alloc), which means the CPU can read/write directly
 * via [buffer contents] — no encode/copy needed.
 * ====================================================================== */

/* IO Unit — write a float32 array into a buffer at offset 0.
 * `vals` is a Lean FloatArray (double-precision, internally); we
 * convert each element to float32 and write into the buffer. */
lean_obj_res lean_theograd_metal_buffer_write_f32(
        uint64_t buf_ptr, b_lean_obj_arg vals, lean_object* w) {
    (void)w;
    void* dst = theograd_metal_buffer_contents((void*)(uintptr_t)buf_ptr);
    if (!dst) return lean_io_result_mk_ok(lean_box(0));
    size_t n = lean_sarray_size(vals);
    const double* src = lean_float_array_cptr(vals);
    float* fdst = (float*)dst;
    for (size_t i = 0; i < n; i++) {
        fdst[i] = (float)src[i];
    }
    return lean_io_result_mk_ok(lean_box(0));
}

/* IO Float — read a single float32 element at index i from a buffer. */
lean_obj_res lean_theograd_metal_buffer_read_f32(
        uint64_t buf_ptr, size_t index, lean_object* w) {
    (void)w;
    void* src = theograd_metal_buffer_contents((void*)(uintptr_t)buf_ptr);
    double val = 0.0;
    if (src) {
        const float* fsrc = (const float*)src;
        val = (double)fsrc[index];
    }
    return lean_io_result_mk_ok(lean_box_float(val));
}

/* IO Unit — copy bytes from a Lean ByteArray into a buffer at offset 0.
 *
 * Used for arbitrary-dtype host→device transfer (notably bf16, which
 * has no numpy dtype but lives just fine inside a ByteArray on the
 * Lean side). The L5.b gate writes captured bf16 bytes through this
 * path, then byte-compares the output against a captured fixture. */
lean_obj_res lean_theograd_metal_buffer_write_bytes(
        uint64_t buf_ptr, b_lean_obj_arg bytes, lean_object* w) {
    (void)w;
    void* dst = theograd_metal_buffer_contents((void*)(uintptr_t)buf_ptr);
    if (!dst) return lean_io_result_mk_ok(lean_box(0));
    size_t n = lean_sarray_size(bytes);
    const uint8_t* src = lean_sarray_cptr((lean_object*)bytes);
    memcpy(dst, src, n);
    return lean_io_result_mk_ok(lean_box(0));
}

/* IO ByteArray — copy n_bytes from a buffer's contents into a freshly
 * allocated Lean ByteArray. */
lean_obj_res lean_theograd_metal_buffer_read_bytes(
        uint64_t buf_ptr, size_t n_bytes, lean_object* w) {
    (void)w;
    void* src = theograd_metal_buffer_contents((void*)(uintptr_t)buf_ptr);
    lean_obj_res arr = lean_alloc_sarray(1, n_bytes, n_bytes);
    uint8_t* dst = lean_sarray_cptr(arr);
    if (src) {
        memcpy(dst, src, n_bytes);
    } else {
        memset(dst, 0, n_bytes);
    }
    return lean_io_result_mk_ok(arr);
}

/* IO Int32 — synchronous dispatch.
 *
 * Lean signature:
 *   metalDispatch : UInt64 → @& String → @& Array UInt64 →
 *                   UInt32 × UInt32 × UInt32 →   -- grid (gx, gy, gz)
 *                   UInt32 × UInt32 × UInt32 →   -- threadgroup (lx, ly, lz)
 *                   IO Int32
 *
 * Returns 0 on success or negative error per metal_alloc.m's contract.
 *
 * To keep the C signature simple, the 6 launch dims pack into 6
 * positional size_t args at the C level. */
lean_obj_res lean_theograd_metal_dispatch(
        uint64_t library_ptr,
        b_lean_obj_arg fn_name,
        b_lean_obj_arg buffers_arr,
        size_t gx, size_t gy, size_t gz,
        size_t lx, size_t ly, size_t lz,
        lean_object* w) {
    (void)w;
    const char* fn_c = lean_string_cstr(fn_name);
    size_t n = lean_array_size(buffers_arr);
    void** bufs = (void**)alloca(n * sizeof(void*));
    for (size_t i = 0; i < n; i++) {
        uint64_t u = lean_unbox_uint64(lean_array_get_core(buffers_arr, i));
        bufs[i] = (void*)(uintptr_t)u;
    }
    int rc = theograd_metal_dispatch((void*)(uintptr_t)library_ptr,
                                      fn_c, bufs, n,
                                      gx, gy, gz, lx, ly, lz);
    return lean_io_result_mk_ok(lean_box_uint32((uint32_t)(int32_t)rc));
}
