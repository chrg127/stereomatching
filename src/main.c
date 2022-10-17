#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define SIZE 128
#define NUM_SHIFTS 30
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

int idx(int x, int y, int w)
{
    x = (x + w) % w;
    y = (y + w) % w;
    return y * w + x;
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    return p;
}

// convert values from 0..256 to 0.0..1.0
double *convert_image(u8 *data, int width, int height)
{
    double *newdata = xmalloc(sizeof(double) * width * height);
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
    out->data = convert_image(imgdata, out->width, out->height);
    return 0;
}

void write_image_data(double *data, int width, int height, const char *name, int number)
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

void write_image(Image *im, const char *name, int number)
{
    write_image_data(im->data, im->width, im->height, name, number);
}

void free_image(Image *im)
{
    stbi_image_free(im->data);
}

int find_edges_left_right(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[idx(x-1, y-1, width)];
    double v2 = brightness[idx(x-1, y  , width)];
    double v3 = brightness[idx(x-1, y+1, width)];
    double v4 = brightness[idx(x+1, y-1, width)];
    double v5 = brightness[idx(x+1, y  , width)];
    double v6 = brightness[idx(x+1, y+1, width)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_top_bottom(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[idx(x-1, y-1, width)];
    double v2 = brightness[idx(x  , y-1, width)];
    double v3 = brightness[idx(x+1, y-1, width)];
    double v4 = brightness[idx(x-1, y+1, width)];
    double v5 = brightness[idx(x  , y+1, width)];
    double v6 = brightness[idx(x+1, y+1, width)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_upleft_downright(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[idx(x-1, y-1, width)];
    double v2 = brightness[idx(x  , y-1, width)];
    double v3 = brightness[idx(x-1, y  , width)];
    double v4 = brightness[idx(x+1, y  , width)];
    double v5 = brightness[idx(x  , y+1, width)];
    double v6 = brightness[idx(x+1, y+1, width)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_downleft_upright(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[idx(x-1, y+1, width)];
    double v2 = brightness[idx(x  , y+1, width)];
    double v3 = brightness[idx(x-1, y  , width)];
    double v4 = brightness[idx(x  , y-1, width)];
    double v5 = brightness[idx(x+1, y-1, width)];
    double v6 = brightness[idx(x+1, y  , width)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

void find_all_edges(double *brightness, int width, int height, double threshold, double *out)
{
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            if (find_edges_left_right(brightness, width, x, y, threshold)
             || find_edges_top_bottom(brightness, width, x, y, threshold)
             || find_edges_upleft_downright(brightness, width, x, y, threshold)
             || find_edges_downleft_upright(brightness, width, x, y, threshold))
                out[idx(x, y, width)] = 0.0;
            else
                out[idx(x, y, width)] = 1.0;
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

double *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        matches[i] = xmalloc(sizeof(matches[0]) * width * height);
}

void fillup_matches(double *left_edges, double *right_edges, int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x,   y, width),
                    shift = idx(x+i, y, width);
                // ^ the +i accomplishes the sliding process
                matches[i][index] = left_edges[index] == right_edges[shift] ? 0.0 : 1.0;
            }
        }
    }
}

int *addup_pixels_in_square(double *p, int width, int height, int square_width)
{
    int half = square_width / 2;
    int *total = malloc(sizeof(int) * width * height);
    for (int sy = 0; sy < square_width; sy++) {
        for (int sx = 0; sx < square_width; sx++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int cur = idx(x, y, width);
                    int rel = idx(x + sx - half,
                                  y + sy - half, width);
                    total[cur] += p[rel];
                }
            }
        }
    }
    return total;
}

int *scores[NUM_SHIFTS];

bool has_match(int *p, int width, int height)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            if (p[idx(x, y, width)] == 1)
                return true;
    return false;
}

void fillup_scores(int width, int height, int square_width)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int *sums = addup_pixels_in_square(matches[i], width, height, square_width);
        scores[i] = has_match(matches[i], width, height) ? sums : NULL;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "usage: stereomatch [image1] [image2] [threshold = 0.15]\n");
        return 1;
    }

    Image first, second;
    if (read_image(argv[1], &first))
        return 1;
    if (read_image(argv[2], &second))
        return 1;
    if (first.width != second.width || first.height != second.height) {
        fprintf(stderr, "error: the two images must have equal width and height\n");
        return 1;
    }

    double threshold = 0.15;
    if (argc >= 4) {
        char *endptr;
        threshold = strtod(argv[3], &endptr);
        if (threshold == 0 && endptr == argv[3]) {
            fprintf(stderr, "error: threshold must be a number\n");
            return 1;
        }
    }

    /* first step: find edges in both images */
    double *first_edges = xmalloc(sizeof(double) * first.width * first.height);
    memset(first_edges, 0, sizeof(double) * first.width * first.height);
    find_all_edges(first.data, first.width, first.height, threshold, first_edges);
    write_image_data(first_edges, first.width, first.height, "edges", 1);

    double *second_edges = xmalloc(sizeof(double) * first.width * first.height);
    memset(second_edges, 0, sizeof(double) * first.width * first.height);
    find_all_edges(second.data, second.width, second.height, threshold, second_edges);
    write_image_data(second_edges, second.width, second.height, "edges", 2);

    /* second step: match edges between images */
    allocate_matches(first.width, first.height);
    fillup_matches(first.data, second.data, first.width, first.height);
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image_data(matches[i], first.width, first.height, "matches", i);

    free(first_edges);
    free(second_edges);
    free_image(&first);
    free_image(&second);
    return 0;
}
