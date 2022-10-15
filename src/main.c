#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SIZE 128
#define IDX(x, y, w) ((y)*(w)+(x))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define CLAMP(x, mi, ma) (MIN(MAX(x, mi), ma))

typedef uint8_t  u8;
typedef uint32_t u32;
typedef int8_t   i8;
typedef int32_t  i32;

typedef struct {
    double *data;
    int width, height;
} Image;

// convert values from 0...256 to 0.0 ... 1.0
double *convert(u8 *data, int width, int height)
{
    double *newdata = malloc(sizeof(double) * width * height);
    for (int i = 0; i < width * height; i++)
        newdata[i] = data[i] / 256.0;
    return newdata;
}

int read_image(const char *name, Image *out)
{
    int channels = 0;
    u8 *imgdata = stbi_load(name, &out->width, &out->height, &channels, 0);
    if (!imgdata) {
        fprintf(stderr, "error reading image:");
        perror("");
        return 1;
    }
    if (channels != 1) {
        fprintf(stderr, "error reading image: wrong number of channels (%d) (image must be grayscale)", channels);
        return 1;
    }
    out->data = convert(imgdata, out->width, out->height);
    return 0;
}

void write_image(double *data, int width, int height, const char *name, int number)
{
    char filename[1000];
    snprintf(filename, sizeof(filename), "%s%d.ppm", name, number);
    FILE *f = fopen(filename, "w");
    if (!f)
        return;
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        int v = (int) (data[i] * 255.0);
        fprintf(f, "%d %d %d\n", v, v, v);
    }
}

void write_image_u8(u8 *data, int width, int height, const char *name, int number)
{
    char filename[1000];
    snprintf(filename, sizeof(filename), "%s%d.ppm", name, number);
    FILE *f = fopen(filename, "w");
    if (!f)
        return;
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        u8 v = data[i];
        fprintf(f, "%d %d %d\n", v, v, v);
    }
}

int find_edges_left_right(double *brightness, int size, int x, int y, double threshold)
{
    double v1 = brightness[IDX(x-1, y-1, size)];
    double v2 = brightness[IDX(x-1, y  , size)];
    double v3 = brightness[IDX(x-1, y+1, size)];
    double v4 = brightness[IDX(x+1, y-1, size)];
    double v5 = brightness[IDX(x+1, y  , size)];
    double v6 = brightness[IDX(x+1, y+1, size)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_top_bottom(double *brightness, int size, int x, int y, double threshold)
{
    double v1 = brightness[IDX(x-1, y-1, size)];
    double v2 = brightness[IDX(x  , y-1, size)];
    double v3 = brightness[IDX(x+1, y-1, size)];
    double v4 = brightness[IDX(x-1, y+1, size)];
    double v5 = brightness[IDX(x  , y+1, size)];
    double v6 = brightness[IDX(x+1, y+1, size)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_upleft_downright(double *brightness, int size, int x, int y, double threshold)
{
    double v1 = brightness[IDX(x-1, y-1, size)];
    double v2 = brightness[IDX(x  , y-1, size)];
    double v3 = brightness[IDX(x-1, y  , size)];
    double v4 = brightness[IDX(x+1, y  , size)];
    double v5 = brightness[IDX(x  , y+1, size)];
    double v6 = brightness[IDX(x+1, y+1, size)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_downleft_upright(double *brightness, int size, int x, int y, double threshold)
{
    double v1 = brightness[IDX(x-1, y+1, size)];
    double v2 = brightness[IDX(x  , y+1, size)];
    double v3 = brightness[IDX(x-1, y  , size)];
    double v4 = brightness[IDX(x  , y-1, size)];
    double v5 = brightness[IDX(x+1, y-1, size)];
    double v6 = brightness[IDX(x+1, y  , size)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

void find_all_edges(double *brightness, double *out, double threshold, int size, int height)
{
    for (int x = 1; x < size-1; x++) {
        for (int y = 1; y < height-1; y++) {
            if (find_edges_left_right(brightness, size, x, y, threshold)
             || find_edges_top_bottom(brightness, size, x, y, threshold)
             || find_edges_upleft_downright(brightness, size, x, y, threshold)
             || find_edges_downleft_upright(brightness, size, x, y, threshold))
                out[IDX(x, y, size)] = 0.0;
            else
                out[IDX(x, y, size)] = 1.0;
        }
    }
}

int black_pixels(double *im, int width, int height)
{
    int count = 0;
    for (int i = 0; i < width * height; i++) {
        if (im[i] == 0)
            count++;
    }
    return count;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "usage: stereomatch [image1] [image2]\n");
        return 1;
    }

    Image first, second;
    if (read_image(argv[1], &first))
        return 1;
    if (read_image(argv[2], &second))
        return 1;
    // if (first.width != second.width || first.height != second.height) {
    //     fprintf(stderr, "error: the two images must have equal width and height\n");
    //     return 1;
    // }
    // write_image(first.data, first.width, first.height, "first", 0);

    double *out = malloc(sizeof(double) * first.width * first.height);
    double ts[] = { 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    for (size_t i = 0; i < sizeof(ts)/sizeof(ts[0]); i++) {
        double t = ts[i];
        memset(out, 0, sizeof(double) * first.width * first.height);
        find_all_edges(first.data, out, t, first.width, first.height);
        int blacks = black_pixels(out, first.width, first.height);
        printf("%d: %d blacks (%d%%)\n", i, blacks, blacks * 100 / (first.width*first.height));
        write_image(out, first.width, first.height, "edges", i);
    }
    free(out);

    printf("ok\n");
    return 0;
}
