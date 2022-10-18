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



// utility functions

int idx(int x, int y, int w)
{
    x = (x + w) % w;
    y = (y + w) % w;
    return y * w + x;
}

void *xmalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (!p) {
        fprintf(stderr, "error: out of memory\n");
        exit(1);
    }
    return p;
}



// image I/O

// convert values from 0..256 to 0.0..1.0
double *convert_image(u8 *data, int width, int height)
{
    double *newdata = xmalloc(width * height, sizeof(double));
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

// free images opened by read_image
void free_image(Image *im)
{
    stbi_image_free(im->data);
}

// write a grayscale image. get_data(data, i) returns a single pixel value from the image
void write_image(void *data, int width, int height, const char *name, int number, int (*get_data)(void *, int))
{
    char filename[1000];
    snprintf(filename, sizeof(filename), "%s%d.ppm", name, number);
    FILE *f = fopen(filename, "w");
    if (!f)
        return;
    fprintf(f, "P3\n%d %d\n255\n", width, height);
    for (int i = 0; i < width * height; i++) {
        int v = get_data(data, i);
        fprintf(f, "%d %d %d\n", v, v, v);
    }
}

// used for grayscale images where each pixel is in a 0.0..1.0 range, where 0.0 is black
int get_data_grayscale(void *data, int i)
{
    double *p = (double *) data;
    return (int) (p[i] * 255.0);
}

// used for 'binary' images, i.e. images composed only of 0s and 1s, where 0 is white (and NOT black)
int get_data_binary(void *data, int i)
{
    u8 *p = (u8 *) data;
    return (int) (p[i] == 1 ? 0 : 255);
}



// the actual algorithm

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

void find_all_edges(double *brightness, int width, int height, double threshold, u8 *out)
{
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            out[idx(x, y, width)] =
                find_edges_left_right(brightness, width, x, y, threshold)
             || find_edges_top_bottom(brightness, width, x, y, threshold)
             || find_edges_upleft_downright(brightness, width, x, y, threshold)
             || find_edges_downleft_upright(brightness, width, x, y, threshold);
        }
    }
}

// a WxH size array used to keep matches
u8 *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        matches[i] = xmalloc(width * height, sizeof(matches[0]));
}

void fillup_matches(u8 *left_edges, u8 *right_edges, int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x,   y, width),
                    shift = idx(x+i, y, width);
                // ^ the +i accomplishes the sliding process
                matches[i][index] = left_edges[index] == right_edges[shift];
            }
        }
    }
}

/*
 * for each pixel in parallel:
 *
 * *.. .*. ..* ... ... ... ... ... ...
 * ... ... ... *.. .*. ..* ... ... ...
 * ... ... ... ... ... ... *.. .*. ..*
 *
 * where the considered pixel is at the center and square_width = 3
 * pixels must be a binary image.
 */
i32 *addup_pixels_in_square(u8 *pixels, int width, int height, int square_width)
{
    int half = square_width / 2;
    i32 *total = xmalloc(width * height, sizeof(i32));
    for (int sy = 0; sy < square_width; sy++) {
        for (int sx = 0; sx < square_width; sx++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int cur = idx(x, y, width);
                    int rel = idx(x + sx - half,
                                  y + sy - half, width);
                    total[cur] += (i32) pixels[rel];
                }
            }
        }
    }
    return total;
}

// a WxH size array used to keep scores
i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        scores[i] = xmalloc(width * height, sizeof(scores[0]));
}

void fillup_scores(int width, int height, int square_width)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        i32 *sums = addup_pixels_in_square(matches[i], width, height, square_width);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x, y, width);
                // record a score whenever there was a match-up
                if (matches[i][index] == 1)
                    scores[i][index] = sums[index];
            }
        }
    }
}

void find_shifts_of_highest_scoring(int width, int height)
{
    i32 *best_scores    = xmalloc(width * height, sizeof(int));
    i32 *winning_shifts = xmalloc(width * height, sizeof(int));
    // the following loop makes sure that each pixel in the best_scores
    // image contains the maximum score found at any shift.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x, y, width);
                if (scores[i][index] > best_scores[index])
                    best_scores[index] = scores[i][index];
            }
        }
    }
    // the following loop records a 'winning' shift at every pixel
    // whose score is the best.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x, y, width);
                if (scores[i][index] == best_scores[index])
                    winning_shifts[index] = i+1;
            }
        }
    }
    return winning_shifts;
}



// other random stuff

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
    u8 *first_edges = xmalloc(first.width * first.height, sizeof(u8));
    find_all_edges(first.data, first.width, first.height, threshold, first_edges);
    write_image(first_edges, first.width, first.height, "edges", 1, get_data_binary);

    u8 *second_edges = xmalloc(first.width * first.height, sizeof(u8));
    find_all_edges(second.data, second.width, second.height, threshold, second_edges);
    write_image(second_edges, second.width, second.height, "edges", 2, get_data_binary);

    /* second step: match edges between images */
    allocate_matches(first.width, first.height);
    fillup_matches(first_edges, second_edges, first.width, first.height);
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image(matches[i], first.width, first.height, "matches", i, get_data_binary);

    free(first_edges);
    free(second_edges);
    free_image(&first);
    free_image(&second);
    return 0;
}
