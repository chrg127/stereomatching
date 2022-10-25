CC := gcc
CFLAGS := -g -Wall -Wextra -pedantic -std=c11
CUFLAGS := -g
LDLIBS := -lm
name := stereomatch
name_par := stereopar

all: $(name) $(name_par)

$(name): src/stereo.o
	gcc $< -o $@ $(LDLIBS)

$(name_par): src/stereo.cu
	nvcc $(CUFLAGS) $< -o $@ $(LDLIBS)

.PHONY: clean

clean:
	-rm $(name) $(name_par) *.ppm src/*.o
