LEAN_PREFIX  := $(shell lean --print-prefix)
LEAN_INCLUDE := $(LEAN_PREFIX)/include

.PHONY: all

all: build/hip_probe.o

build/hip_probe.o: hip_probe.c
	@mkdir -p build
	clang -O2 -fPIC -Wall -Wextra -I$(LEAN_INCLUDE) -c -o $@ $<
