#include <stdint.h>
#include <dlfcn.h>
#include <lean/lean.h>

/* Header-free HIP discovery.  This bridge owns no Tensor semantics and emits
   only a count-wire fact: 0 missing library, 1 missing symbol, 2 query error,
   3 zero devices, and 256+n for n>0 devices. */
static uint64_t tgrad_hip_probe_raw(void) {
  static const char *candidates[] = {
    "libamdhip64.so", "libamdhip64.so.6", "libamdhip64.dylib"
  };
  void *library = 0;
  for (unsigned i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
    library = dlopen(candidates[i], RTLD_NOW | RTLD_LOCAL);
    if (library != 0) break;
  }
  if (library == 0) return 0;

  typedef int (*hip_get_device_count_fn)(int *);
  hip_get_device_count_fn get_device_count =
    (hip_get_device_count_fn)dlsym(library, "hipGetDeviceCount");
  if (get_device_count == 0) {
    dlclose(library);
    return 1;
  }

  int count = 0;
  int result = get_device_count(&count);
  dlclose(library);
  if (result != 0) return 2;
  if (count <= 0) return 3;
  return 256u + (uint64_t)count;
}

lean_obj_res lean_tgrad_hip_probe(lean_object *world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint64(tgrad_hip_probe_raw()));
}
