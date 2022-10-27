name := stereomatch
name_par := stereopar
_objs 		 := image.c.o
_objs_serial := stereo.c.o
_objs_par 	 := stereo.cu.o image.cu.o
CC := gcc
NVCC := nvcc
CFLAGS := -g -Wall -Wextra -pedantic -std=c11
NVFLAGS := -g
flags_deps = -MMD -MP -MF $(@:.o=.d)
LDLIBS := -lm
outdir := out

VPATH := src
objs 		:= $(patsubst %,$(outdir)/%,$(_objs))
objs_serial := $(patsubst %,$(outdir)/%,$(_objs_serial))
objs_par 	:= $(patsubst %,$(outdir)/%,$(_objs_par))

all: $(outdir) $(outdir)/$(name) $(outdir)/$(name_par)

$(outdir):
	mkdir -p $(outdir)

$(outdir)/$(name): $(objs) $(objs_serial)
	$(CC) $(objs) $(objs_serial) -o $@ $(LDLIBS)

$(outdir)/$(name_par): $(objs) $(objs_par)
	$(NVCC) $(objs) $(objs_par) -o $@ $(LDLIBS)

-include $(outdir)/*.d

$(outdir)/%.c.o: %.c
	$(CC) $(CFLAGS) $(flags_deps) -c $< -o $@

$(outdir)/%.cu.o: %.cu
	$(NVCC) $(NVFLAGS) $(flags_deps) -c $< -o $@

.PHONY: clean

clean:
	-rm -r *.ppm $(outdir)
