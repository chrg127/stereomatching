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

int read_image(const char *name, Image *out);
void write_image(void *data, int width, int height, int ghost_size, ImageType type, const char *name, int number);

#ifdef __NVCC__
void write_image_from_gpu(void *data, int width, int height, ImageType type, const char *name, int number);
#endif

#ifdef __NVCC__
}
#endif

#endif
