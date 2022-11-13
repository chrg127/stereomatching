#include "image.h"

#include <stdio.h>
#include "ghost.h"
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

static int get_image_value(void *p, int x, int y, int width, int ghost_size, ImageType type)
{
    switch (type) {
    case IMTYPE_BINARY:     return (int) (((u8 *)p)[IGX(x, y, width, ghost_size)] == 1 ? 0 : 255);
    case IMTYPE_GRAY_FLOAT: return (int) (((double *)p)[IGX(x, y, width, ghost_size)] * 255.0);
    case IMTYPE_GRAY_INT:   return (int) (((i32 *)p)[IGX(x, y, width, ghost_size)]);
    default:                return 0;
    }
}

char *make_filename(const char *name, ImageProgramType type, int number)
{
    char *filename = ALLOCATE(char, 1024);
    // in debug mode create images in another directory.
    // this is for the diff.sh script to work.
#ifdef DEBUG
    switch (type) {
    case SER:      snprintf(filename, sizeof(char) * 1024, "%s/%s-%d.ppm", "ser",   name, number); break;
    case PAR:      snprintf(filename, sizeof(char) * 1024, "%s/%s-%d.ppm", "par",   name, number); break;
    case SERGHOST: snprintf(filename, sizeof(char) * 1024, "%s/%s-%d.ppm", "sergh", name, number); break;
    case PARGHOST: snprintf(filename, sizeof(char) * 1024, "%s/%s-%d.ppm", "pargh", name, number); break;
    }
#else
    snprintf(filename, sizeof(char) * 1024, "%s-%d.ppm", name, number); break;
#endif
    return filename;
}

// writes a grayscale image.
void write_image(void *data, int width, int height, int ghost_size, ImageType type, char *filename)
{
#ifndef NO_WRITES
    FILE *f = fopen(filename, "w");
    free(filename);
    if (!f)
        return;
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int v = get_image_value(data, x, y, width, ghost_size, type);
            fprintf(f, "%d %d %d\n", v, v, v);
        }
    }
#endif
}
