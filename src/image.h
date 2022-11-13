#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include "util.h"

#ifdef __NVCC__
extern "C" {
#endif

typedef struct {
    double *data;
    int width, height;
} Image;

typedef enum ImageType {
    IMTYPE_BINARY,      // an image with only 0s for white and 1s for black
    IMTYPE_GRAY_FLOAT,  // an image where each pixel is a float between 0..1
    IMTYPE_GRAY_INT,    // an image where each pixel is an integer from 0 to 255
} ImageType;

typedef enum ProgramType {
    SER = 0, PAR, SERGHOST, PARGHOST,
} ImageProgramType;

int read_image(const char *name, Image *out);
char *make_filename(const char *name, ImageProgramType type, int number);
void write_image(void *data, int width, int height, int ghost_size, ImageType type, char *filename);

#ifdef __NVCC__
void write_gpu_image(void *data, int width, int height, int ghost_size, ImageType type, char *name);
#endif

#ifdef __NVCC__
}
#endif

#endif
