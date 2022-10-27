name := stereomatch
name_par := stereopar
_objs 		 := image.c.o
_objs_serial := stereo.c.o
_objs_par 	 := stereo.cu.o
CC := gcc
NVCC := nvcc
CFLAGS := -g -Wall -Wextra -pedantic -std=c11
CUFLAGS := -g
flags_deps = -MMD -MP -MF $(@:.o=.d)
LDLIBS := -lm
outdir := out

VPATH := src
objs 		:= $(patsubst %,$(outdir)/%,$(_objs))
objs_serial := $(patsubst %,$(outdir)/%,$(_objs_serial))
objs_par 	:= $(patsubst %,$(outdir)/%,$(_objs_par))

all: $(outdir) $(outdir)/$(name) #$(name_par)

$(outdir):
	mkdir -p $(outdir)

$(outdir)/$(name): $(objs) $(objs_serial)
	$(info Linking $@ ...)
	$(CC) $(objs) $(objs_serial) -o $@ $(LDLIBS)

# $(outdir)/$(name_par): src/stereo.cu
# 	nvcc $(CUFLAGS) $< -o $@ $(LDLIBS)

-include $(outdir)/*.d

$(outdir)/%.c.o: %.c
	$(info Compiling $< ...)
	@$(CC) $(CFLAGS) $(flags_deps) -c $< -o $@

$(outdir)/%.c.o: %.c
	$(info Compiling $< ...)
	@$(CC) $(CFLAGS) $(flags_deps) -c $< -o $@

.PHONY: clean

clean:
	-rm $(name) $(name_par) *.ppm src/*.o
