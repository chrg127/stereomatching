name 		   := stereomatch
name_par 	   := stereopar
name_ghost     := stereomatch-ghost
name_ghost_par := stereopar-ghost

build 		 := debug
CC 			 := gcc
NVCC 		 := nvcc
CFLAGS 		 := -g -Wall -Wextra -pedantic -std=gnu11 -Wno-unused-parameter
NVFLAGS 	 :=
flags_deps   = -MMD -MP -MF $(@:.o=.d)
LDLIBS 		 := -lm

ifeq ($(build),debug)
    outdir := debug
    CFLAGS += -g -DDEBUG
	NVFLAGS += -g -DDEBUG
else ifeq ($(build),timing)
    outdir := timing
    CFLAGS += -O3 -DNO_WRITE
	NVFLAGS := -O3 -DNO_WRITE
else
	$(error error: invalid value for build)
endif

VPATH 		       := src
_objs 		  	   := image.c.o
_objs_serial 	   := stereo.c.o
_objs_par 	 	   := stereo.cu.o image.cu.o util.cu.o
_objs_serial_ghost := stereo-ghost.c.o
_objs_par_ghost    := stereo-ghost.cu.o image.cu.o util.cu.o
objs 		 	   := $(patsubst %,$(outdir)/%,$(_objs))
objs_serial  	   := $(patsubst %,$(outdir)/%,$(_objs_serial))
objs_par 	 	   := $(patsubst %,$(outdir)/%,$(_objs_par))
objs_serial_ghost  := $(patsubst %,$(outdir)/%,$(_objs_serial_ghost))
objs_par_ghost 	   := $(patsubst %,$(outdir)/%,$(_objs_par_ghost))

all: $(outdir) $(outdir)/$(name) $(outdir)/$(name_par) $(outdir)/$(name_ghost) $(outdir)/$(name_ghost_par)

$(outdir):
	mkdir -p $(outdir)

$(outdir)/$(name): $(objs) $(objs_serial)
	$(CC) $(objs) $(objs_serial) -o $@ $(LDLIBS)

$(outdir)/$(name_par): $(objs) $(objs_par)
	$(NVCC) $(objs) $(objs_par) -o $@ $(LDLIBS)

$(outdir)/$(name_ghost): $(objs) $(objs_serial_ghost)
	$(CC) $(objs) $(objs_serial_ghost) -o $@ $(LDLIBS)

$(outdir)/$(name_ghost_par): $(objs) $(objs_par_ghost)
	$(NVCC) $(objs) $(objs_par) -o $@ $(LDLIBS)

-include $(outdir)/*.d

$(outdir)/%.c.o: %.c
	$(CC) $(CFLAGS) $(flags_deps) -c $< -o $@

$(outdir)/%.cu.o: %.cu
	$(NVCC) $(NVFLAGS) $(flags_deps) -c $< -o $@

.PHONY: clean

clean:
	-rm -r *.ppm debug timing
