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
#define SWAP(x, y, T)   \
    do {                \
        T tmp = (x);    \
        (x) = (y);      \
        (y) = tmp;      \
    } while (0)

#define DEFAULT_THRESHOLD 0.15
#define DEFAULT_SQUARE_WIDTH 5
#define DEFAULT_TIMES 32
#define DEFAULT_LINES 10

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

double parse_double(const char *s, double *n)
{
    char *endptr;
    *n = strtod(s, &endptr);
    return *n == 0 && endptr == s;
}

int parse_int(const char *s, int *n)
{
    char *endptr;
    *n = strtol(s, &endptr, 0);
    return *n == 0 && endptr == s;
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
    stbi_image_free(imgdata);
    return 0;
}

typedef enum ImageType {
    IMTYPE_BINARY,      // an image with only 0s for white and 1s for black
    IMTYPE_GRAY_FLOAT,  // an image where each pixel is a floating point value between 0..1
    IMTYPE_GRAY_INT,    // an image where each pixel is a value from 0 to 255
} ImageType;

int get_image_value(void *p, int i, ImageType type)
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

void find_all_edges(double *brightness, int width, int height, double threshold, u8 *edges)
{
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            edges[idx(x, y, width)] =
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
        write_image(matches[i], width, height, IMTYPE_BINARY, "matches", i);
    }
}

// a WxH size array used to keep scores
i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        scores[i] = xmalloc(width * height, sizeof(scores[0]));
}

/*
 * for each pixel in parallel:
 *
 * *.. .*. ..* ... ... ... ... ... ...
 * ... ... ... *.. .*. ..* ... ... ...
 * ... ... ... ... ... ... *.. .*. ..*
 *
 * (where the considered pixel is at the center and square_width = 3)
 * pixels must be a binary image.
 */
void addup_pixels_in_square(u8 *pixels, int width, int height, int square_width, i32 *total)
{
    int half = square_width / 2;
    memset(total, 0, sizeof(total[0]) * width * height);
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
}

void fillup_scores(int width, int height, int square_width, i32 *sum)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        addup_pixels_in_square(matches[i], width, height, square_width, sum);
        write_image(sum, width, height, IMTYPE_GRAY_INT, "score_all", i);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = idx(x, y, width);
                // record a score whenever there was a match-up
                if (matches[i][index] == 1)
                    scores[i][index] = sum[index];
            }
        }
        write_image(scores[i], width, height, IMTYPE_GRAY_INT, "score_edges", i);
    }
}

// this function computes the web of known shifts. recall that
// the shift at each pixel corresponds directly to the elevation.
void find_highest_scoring_shifts(i32 *best_scores, i32 *winning_shifts, int width, int height)
{
    memset(best_scores,    0, sizeof(best_scores[0])    * width * height);
    memset(winning_shifts, 0, sizeof(winning_shifts[0]) * width * height);
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
    write_image(best_scores, width, height, IMTYPE_GRAY_INT, "best_scores", 0);
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
}

i32 *fill_web_holes(i32 *web, int width, int height, int times)
{
    // each time though the loop, every pixel not on the web (i.e., every pixel that is not
    // zero to begin with) takes on the average elevation of its four neighbors. therefore,
    // the web pixels gradually "spread" their elevations across the holes, while they
    // themselves remain unchanged.
    i32 *tmp = xmalloc(width * height, sizeof(web[0]));
    memcpy(tmp, web, sizeof(web[0]) * width * height);
    for (int i = 0; i < times; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (tmp[idx(x, y, width)] == 0) {
                    web[idx(x, y, width)] =
                        (tmp[idx(x+1, y,   width)]
                       + tmp[idx(x,   y+1, width)]
                       + tmp[idx(x-1, y,   width)]
                       + tmp[idx(x,   y-1, width)]) / 4;
                }
            }
        }
        SWAP(web, tmp, i32 *);
    }
    free(tmp);
    return web;
}

i32 image_max(i32 *im, int width, int height)
{
    i32 max = 0;
    for (int i = 0; i < width*height; i++)
        max = MAX(im[i], max);
    return max;
}

i32 image_min(i32 *im, int width, int height)
{
    i32 min = 0;
    for (int i = 0; i < width*height; i++)
        min = MIN(im[i], min);
    return min;
}

void draw_contour_map(i32 *web, int width, int height, int num_lines, u8 *image_output)
{
    // the idea is to divide the whole range of elevations into a number of intervals,
    // then to draw a contour line at every interval.
    i32 max_elevation = image_max(web, width, height),
        min_elevation = image_min(web, width, height),
        range         = max_elevation - min_elevation,
        interval      = range / num_lines;
    // now the variable 'interval' tells us how many elevations, or shifts, to skip between
    // contour lines.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image_output[idx(x, y, width)] =
                ((web[idx(x, y, width)] - min_elevation) % interval) == 0;
        }
    }
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
        fprintf(stderr, "usage: stereomatch [image 1] [image 2] [threshold = %g] "
                        "[square_width = %d] [times = %d] [lines = %d]\n",
                        DEFAULT_THRESHOLD, DEFAULT_SQUARE_WIDTH, DEFAULT_TIMES, DEFAULT_LINES);
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

    double threshold = DEFAULT_THRESHOLD;
    if (argc >= 4 && parse_double(argv[3], &threshold)) {
        fprintf(stderr, "error: threshold must be a number\n");
        return 1;
    }

    int square_width = DEFAULT_SQUARE_WIDTH;
    if (argc >= 5 && parse_int(argv[4], &square_width)) {
        fprintf(stderr, "error: square_width must be a number\n");
        return 1;
    }

    int times = DEFAULT_TIMES;
    if (argc >= 6 && parse_int(argv[5], &times)) {
        fprintf(stderr, "error: times must be a number\n");
        return 1;
    }

    int lines_to_draw = DEFAULT_LINES;
    if (argc >= 7 && parse_int(argv[6], &lines_to_draw)) {
        fprintf(stderr, "error: lines must be a number\n");
        return 1;
    }

    int width = first.width, height = first.height;

    // first step: find edges in both images
    u8 *first_edges = xmalloc(width * height, sizeof(u8));
    find_all_edges(first.data, width, height, threshold, first_edges);
    write_image(first_edges, width, height, IMTYPE_BINARY, "edges", 1);

    u8 *second_edges = xmalloc(width * height, sizeof(u8));
    find_all_edges(second.data, width, height, threshold, second_edges);
    write_image(second_edges, width, height, IMTYPE_BINARY, "edges", 2);

    // second step: match edges between images
    allocate_matches(width, height);
    fillup_matches(first_edges, second_edges, width, height);

    // third step: compute scores for each pixel
    i32 *buf            = xmalloc(width * height, sizeof(i32)),
        *winning_shifts = xmalloc(width * height, sizeof(i32));
    allocate_scores(width, height);
    fillup_scores(width, height, 5, buf);
    find_highest_scoring_shifts(buf, winning_shifts, width, height);
    write_image(winning_shifts, width, height, IMTYPE_GRAY_INT, "web", 0);

    // fourth step: draw contour lines
    i32 *web = winning_shifts;
    u8 *out = xmalloc(width * height, sizeof(u8));
    web = fill_web_holes(web, width, height, times);
    draw_contour_map(web, width, height, lines_to_draw, out);
    write_image(out, width, height, IMTYPE_BINARY, "output", 0);

    free(first_edges);
    free(second_edges);
    free(first.data);
    free(second.data);
    return 0;
}
