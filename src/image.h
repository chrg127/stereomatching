#ifndef IMAGE_H_INCLUDED
#define IMAGE_H_INCLUDED

#include "util.h"

typedef struct {
    double *data;
    int width, height;
} Image;

typedef enum ImageType {
    IMTYPE_BINARY,      // an image with only 0s for white and 1s for black
    IMTYPE_GRAY_FLOAT,  // an image where each pixel is a float between 0..1
    IMTYPE_GRAY_INT,    // an image where each pixel is an integer from 0 to 255
} ImageType;

double *convert_image(u8 *data, int width, int height);
int read_image(const char *name, Image *out);
int get_image_value(void *p, int i, ImageType type);
void write_image(void *data, int width, int height, ImageType type, const char *name, int number);

#endif
