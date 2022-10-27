#include "image.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// convert values from 0..256 to 0.0..1.0
static double *convert_image(u8 *data, int width, int height)
{
    double *newdata = ALLOCATE(double, width * height);
    for (int i = 0; i < width * height; i++)
        newdata[i] = data[i] / 256.0;
    return newdata;
}

// read image, check if it's grayscale and convert it to doubles in the unit
int read_image(const char *name, Image *out)
{
    int channels = 0;
    u8 *imgdata = stbi_load(name, &out->width, &out->height, &channels, 0);
    if (!imgdata) {
        fprintf(stderr, "error reading image %s:", name);
        perror("");
        return 1;
    }
    if (channels != 1) {
        fprintf(stderr, "error reading image %s: wrong number of channels (%d) "
                        "(image must be grayscale)", name, channels);
        return 1;
    }
    out->data = convert_image(imgdata, out->width, out->height);
    stbi_image_free(imgdata);
    return 0;
}

static int get_image_value(void *p, int i, ImageType type)
{
    switch (type) {
    case IMTYPE_BINARY:     return (int) (((u8 *)p)[i] == 1 ? 0 : 255);
    case IMTYPE_GRAY_FLOAT: return (int) (((double *)p)[i] * 255.0);
    case IMTYPE_GRAY_INT:   return (int) (((i32 *)p)[i]);
    default:                return 0;
    }
}

// writes a grayscale image.
void write_image(void *data, int width, int height, ImageType type, const char *name, int number)
{
    char filename[1000];
    snprintf(filename, sizeof(filename), "%s%d.ppm", name, number);
    FILE *f = fopen(filename, "w");
    if (!f)
        return;
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        int v = get_image_value(data, i, type);
        fprintf(f, "%d %d %d\n", v, v, v);
    }
}
