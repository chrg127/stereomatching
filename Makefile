CC := gcc
CFLAGS := -g -Wall -Wextra -pedantic -std=c11
LDLIBS := -lm
name := stereomatch

all: $(name)

$(name): src/main.o
	gcc $< -o $@ $(LDLIBS)

.PHONY: clean

clean:
	rm $(name) *.ppm src/*.o
