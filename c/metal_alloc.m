/*
 * metal_alloc.m — Objective-C bridge to MTLBuffer alloc/free with a
 * simple LRU pool, lifted from learnings/05_opaque_handle/'s
 * opaque-handle pattern.
 *
 * Used by theograd_phases/13_buffer_allocator (G1 of the bf16 matmul
 * gate). Will also back theograd_phases/14_dispatch_executor (G2).
 *
 * Lifecycle:
 *   - theograd_metal_alloc(size) returns a retained void* (id<MTLBuffer>).
 *     The caller is responsible for balancing with theograd_metal_free.
 *   - theograd_metal_free(ptr, size) caches the buffer if there's room in
 *     the LRU, otherwise releases it (CFBridgingRelease).
 *   - Next alloc(size) with a matching size returns the cached pointer.
 *
 * Compile (Darwin):
 *     clang -c -fobjc-arc -O2 -o metal_alloc.o metal_alloc.m
 * Link (into a Lean binary):
 *     -framework Metal -framework Foundation
 */

#import <Metal/Metal.h>
#include <stdint.h>
#include <stddef.h>

static id<MTLDevice> g_device = nil;

#define LRU_MAX 64
static struct {
    size_t size;
    /* Held as a +1 retained CF reference so the ARC scope of alloc/free
     * doesn't reclaim it while it's parked in the cache. */
    CFTypeRef buf;
} g_lru[LRU_MAX];
static int g_lru_count = 0;

static void ensure_device(void) {
    if (g_device == nil) {
        g_device = MTLCreateSystemDefaultDevice();
    }
}

/* Returns 1 if the system has a default Metal device available, 0
 * otherwise. Lean's Main.lean calls this first to gate the test on
 * hardware availability. */
int theograd_metal_available(void) {
    ensure_device();
    return g_device != nil ? 1 : 0;
}

/* Alloc a buffer of `size` bytes. If a free buffer of the same size is
 * in the LRU cache, that pointer is returned (popped). Otherwise a new
 * MTLBuffer is created via newBufferWithLength:options:.
 *
 * Returns a retained void* representing id<MTLBuffer>. Cast back via
 * (__bridge id<MTLBuffer>)ptr. */
void* theograd_metal_alloc(size_t size) {
    ensure_device();
    if (g_device == nil) return NULL;

    /* LRU pop — most-recently-freed first. */
    for (int i = g_lru_count - 1; i >= 0; i--) {
        if (g_lru[i].size == size) {
            CFTypeRef hit = g_lru[i].buf;
            for (int j = i; j < g_lru_count - 1; j++) g_lru[j] = g_lru[j+1];
            g_lru_count--;
            /* hit is already +1; transfer ownership to caller. */
            return (void*)hit;
        }
    }

    /* No cache hit: alloc fresh. */
    id<MTLBuffer> b = [g_device newBufferWithLength:size
                                            options:MTLResourceStorageModeShared];
    if (b == nil) return NULL;
    /* Retain across the ARC boundary so the caller can hold this past
     * the autorelease pool flush. */
    return (void*)CFBridgingRetain(b);
}

void theograd_metal_free(void* ptr, size_t size) {
    if (!ptr) return;
    if (g_lru_count < LRU_MAX) {
        g_lru[g_lru_count].size = size;
        g_lru[g_lru_count].buf  = (CFTypeRef)ptr;  /* still +1 retained */
        g_lru_count++;
        return;
    }
    /* Cache full — actually release. */
    CFBridgingRelease((CFTypeRef)ptr);  /* -1; ARC reclaims */
}

/* Test helper: total live buffers in the LRU cache (for assertions). */
int theograd_metal_lru_count(void) {
    return g_lru_count;
}

/* Test helper: flush the cache (release all buffers). Used in teardown
 * so successive test runs start from a known state. */
void theograd_metal_lru_flush(void) {
    for (int i = 0; i < g_lru_count; i++) {
        CFBridgingRelease(g_lru[i].buf);
        g_lru[i].buf = NULL;
        g_lru[i].size = 0;
    }
    g_lru_count = 0;
}

/* Return the length of an MTLBuffer (in bytes). Used by tests to
 * sanity-check that the returned pointer points to a real buffer. */
size_t theograd_metal_buffer_length(void* ptr) {
    if (!ptr) return 0;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)ptr;
    return (size_t)[b length];
}

/* ======================================================================
 * G2 (a): MSL source compilation.
 * ====================================================================== */

/* Compile an MSL source string to a Metal library. Returns a retained
 * void* representing id<MTLLibrary> on success, NULL on failure. Caller
 * balances with theograd_metal_library_release.
 *
 * Compilation errors are silenced at the bridge level — the caller can
 * verify success by checking for NULL. Verbose error reporting (for
 * G6's pipeline-composition diagnostics) is a future extension. */
void* theograd_metal_compile(const char* msl_source) {
    ensure_device();
    if (g_device == nil || msl_source == NULL) return NULL;

    NSString* src = [NSString stringWithUTF8String:msl_source];
    NSError* err = nil;
    id<MTLLibrary> lib = [g_device newLibraryWithSource:src
                                                options:nil
                                                  error:&err];
    if (lib == nil) {
        /* Compilation failed — error available in `err` if we ever want
         * to surface it. */
        return NULL;
    }
    return (void*)CFBridgingRetain(lib);
}

/* Release a library returned by theograd_metal_compile. */
void theograd_metal_library_release(void* lib_ptr) {
    if (!lib_ptr) return;
    CFBridgingRelease((CFTypeRef)lib_ptr);
}

/* Return the count of function symbols in a compiled library. Used by
 * G2(a)'s test to verify the source compiled to something usable. */
int theograd_metal_library_function_count(void* lib_ptr) {
    if (!lib_ptr) return -1;
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)lib_ptr;
    return (int)[[lib functionNames] count];
}

/* ======================================================================
 * G2 (b): kernel dispatch + buffer I/O.
 *
 * The command queue is a singleton, owned by this bridge for the
 * process lifetime. Each dispatch builds a fresh command buffer +
 * compute encoder.
 * ====================================================================== */

static id<MTLCommandQueue> g_queue = nil;
static NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* g_pipeline_cache = nil;

static void ensure_queue(void) {
    ensure_device();
    if (g_queue == nil && g_device != nil) {
        g_queue = [g_device newCommandQueue];
    }
    if (g_pipeline_cache == nil) {
        g_pipeline_cache = [NSMutableDictionary dictionary];
    }
}

/* Get a CPU-visible pointer into a buffer's contents. With
 * MTLResourceStorageModeShared (which alloc uses), this is a direct
 * pointer into GPU-visible memory that the CPU can also read/write. */
void* theograd_metal_buffer_contents(void* buf_ptr) {
    if (!buf_ptr) return NULL;
    id<MTLBuffer> b = (__bridge id<MTLBuffer>)buf_ptr;
    return [b contents];
}

/* Synchronously dispatch a kernel.
 *
 * Returns 0 on success, negative error code on failure:
 *   -1: NULL inputs
 *   -2: pipeline creation failed
 *   -3: function not found
 *   -4: command buffer execution error
 *
 * `buffer_indices` is an array of MTLBuffer pointers; each is bound
 * via setBuffer:offset:atIndex: at index i. */
int theograd_metal_dispatch(
        void* library_ptr,
        const char* function_name,
        void* const* buffers, size_t n_buffers,
        size_t gx, size_t gy, size_t gz,
        size_t lx, size_t ly, size_t lz) {
    ensure_queue();
    if (g_device == nil || g_queue == nil) return -1;
    if (!library_ptr || !function_name || !buffers) return -1;

    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)library_ptr;
    NSString* fn = [NSString stringWithUTF8String:function_name];
    NSString* pipe_key = [NSString stringWithFormat:@"%p:%@", library_ptr, fn];
    id<MTLComputePipelineState> pipe = [g_pipeline_cache objectForKey:pipe_key];
    if (pipe == nil) {
        id<MTLFunction> func = [lib newFunctionWithName:fn];
        if (func == nil) return -3;

        NSError* err = nil;
        pipe = [g_device newComputePipelineStateWithFunction:func error:&err];
        if (pipe == nil) return -2;
        [g_pipeline_cache setObject:pipe forKey:pipe_key];
    }

    id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    for (size_t i = 0; i < n_buffers; i++) {
        id<MTLBuffer> b = (__bridge id<MTLBuffer>)buffers[i];
        [enc setBuffer:b offset:0 atIndex:i];
    }
    MTLSize gridSize = MTLSizeMake(gx, gy, gz);
    MTLSize tgSize   = MTLSizeMake(lx, ly, lz);
    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if ([cmd status] != MTLCommandBufferStatusCompleted) return -4;
    return 0;
}
