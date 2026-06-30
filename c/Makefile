# Tgrad C/Obj-C bridge — Metal alloc / compile / dispatch + Python FFI.
#
# Two artifact tiers:
#   1. build/metal_alloc.o + build/metal_alloc_lean.o — linked into
#      tgrad-cli and tgrad-tests via the Lake `moreLinkArgs` clause.
#   2. ../.lake/build/lib/libtgrad.dylib — the Python-facing shared
#      library. Combines build/tgrad_python.o (the Python → Lean
#      trampolines), libtgrad_Tgrad.dylib (the Lean library, built
#      by `lake build`), and Apple's Metal frameworks. Loaded by
#      python/tgrad.py via ctypes.

OS := $(shell uname -s)

CFLAGS := -O2 -fPIC -fobjc-arc -Wall -Wno-deprecated-declarations
ifneq ($(OS),Darwin)
  $(error Tgrad C bridge requires macOS for Metal)
endif

LEAN_PREFIX  := $(shell lean --print-prefix)
LEAN_INCLUDE := $(LEAN_PREFIX)/include
LEAN_LIBDIR  := $(LEAN_PREFIX)/lib/lean
SDK_PATH     ?= $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null || echo /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk)

OBJS    := build/metal_alloc.o build/metal_alloc_lean.o
PY_OBJ  := build/tgrad_python.o
DYLIB   := ../.lake/build/lib/libtgrad.dylib
LEAN_DYLIB := ../.lake/build/lib/libtgrad_Tgrad.dylib
INSTALL_NAME_STAMP := ../.lake/build/lib/.libtgrad_Tgrad.install_name

.PHONY: all clean dylib

all: $(OBJS)

# The Python-facing dylib. Run `lake build` (which produces
# libtgrad_Tgrad.dylib) before invoking this target.
dylib: $(DYLIB)

build/metal_alloc.o: metal_alloc.m
	@mkdir -p build
	clang $(CFLAGS) -c -o $@ $<

build/metal_alloc_lean.o: metal_alloc_lean.c
	@mkdir -p build
	clang $(CFLAGS) -I$(LEAN_INCLUDE) -c -o $@ $<

build/tgrad_python.o: tgrad_python.c
	@mkdir -p build
	clang -O2 -fPIC -Wall -Wno-deprecated-declarations -I$(LEAN_INCLUDE) -c -o $@ $<

# libtgrad.dylib — Python entry point.
#
# Linking strategy: pull in the @[export] symbols from libtgrad_Tgrad.dylib
# (the precompiled Lean library produced by `lake build`), plus the C
# trampolines (this file's compiled .o), plus the Metal Obj-C bridge,
# plus Lean's runtime + Apple's frameworks.
#
# Why -force_load on libtgrad_Tgrad: dyld won't pull symbols from a
# dylib unless something references them; the Lean module's static
# initializers + the @[export] entries must be resolved at load time.
# (The alternative — re-linking every individual tgrad_Tgrad_*.dylib —
# matches what Lean's interpreter does, but is brittle.)
$(INSTALL_NAME_STAMP): $(LEAN_DYLIB)
	install_name_tool -id @rpath/libtgrad_Tgrad.dylib $(LEAN_DYLIB)
	@touch $@

$(DYLIB): $(PY_OBJ) $(OBJS) $(LEAN_DYLIB) $(INSTALL_NAME_STAMP)
	clang -dynamiclib -fPIC -O2 -o $@ \
	  -install_name @rpath/libtgrad.dylib \
	  $(PY_OBJ) build/metal_alloc.o build/metal_alloc_lean.o \
	  -Wl,-force_load,$(LEAN_DYLIB) \
	  -L$(LEAN_LIBDIR) -lleanshared -lLake_shared \
	  -Wl,-rpath,@loader_path \
	  -Wl,-rpath,$(LEAN_LIBDIR) \
	  -isysroot $(SDK_PATH) \
	  -F$(SDK_PATH)/System/Library/Frameworks \
	  -framework Metal -framework Foundation \
	  -framework CoreFoundation -framework CoreGraphics \
	  -lobjc

clean:
	rm -rf build
