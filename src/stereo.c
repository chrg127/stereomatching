#include "util.h"
#include <math.h>
#include "image.h"
#include "ghost.h"

#define NUM_SHIFTS 30
#define DEFAULT_THRESHOLD 0.15
#define DEFAULT_SQUARE_WIDTH 5
#define DEFAULT_TIMES 32
#define DEFAULT_LINES 10
#define MATCHES_GHOST_SIZE 5



// step 1

int find_edges_left_right(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[IGX(x-1, y-1, width, 1)];
    double v2 = brightness[IGX(x-1, y  , width, 1)];
    double v3 = brightness[IGX(x-1, y+1, width, 1)];
    double v4 = brightness[IGX(x+1, y-1, width, 1)];
    double v5 = brightness[IGX(x+1, y  , width, 1)];
    double v6 = brightness[IGX(x+1, y+1, width, 1)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_top_bottom(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[IGX(x-1, y-1, width, 1)];
    double v2 = brightness[IGX(x  , y-1, width, 1)];
    double v3 = brightness[IGX(x+1, y-1, width, 1)];
    double v4 = brightness[IGX(x-1, y+1, width, 1)];
    double v5 = brightness[IGX(x  , y+1, width, 1)];
    double v6 = brightness[IGX(x+1, y+1, width, 1)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_upleft_downright(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[IGX(x-1, y-1, width, 1)];
    double v2 = brightness[IGX(x  , y-1, width, 1)];
    double v3 = brightness[IGX(x-1, y  , width, 1)];
    double v4 = brightness[IGX(x+1, y  , width, 1)];
    double v5 = brightness[IGX(x  , y+1, width, 1)];
    double v6 = brightness[IGX(x+1, y+1, width, 1)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

int find_edges_downleft_upright(double *brightness, int width, int x, int y, double threshold)
{
    double v1 = brightness[IGX(x-1, y+1, width, 1)];
    double v2 = brightness[IGX(x  , y+1, width, 1)];
    double v3 = brightness[IGX(x-1, y  , width, 1)];
    double v4 = brightness[IGX(x  , y-1, width, 1)];
    double v5 = brightness[IGX(x+1, y-1, width, 1)];
    double v6 = brightness[IGX(x+1, y  , width, 1)];
    double avg_left  = (v1 + v2 + v3) / 3.0;
    double avg_right = (v4 + v5 + v6) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall, 0.0, 1.0);
}

void find_all_edges(double *brightness, int width, int height, double threshold, u8 *edges)
{
    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            edges[IGX(x, y, width, 30)] =
                find_edges_left_right(brightness, width, x, y, threshold)
             || find_edges_top_bottom(brightness, width, x, y, threshold)
             || find_edges_upleft_downright(brightness, width, x, y, threshold)
             || find_edges_downleft_upright(brightness, width, x, y, threshold);
        }
    }
}



// step 2

// a WxH size array used to keep matches
u8 *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        matches[i] = ghost_alloc_u8(width, height, MATCHES_GHOST_SIZE, 0);
}

void write_matches(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image(matches[i], width, height, MATCHES_GHOST_SIZE, IMTYPE_BINARY, "matches", i);
}

void free_matches(int width)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        GHOST_FREE(u8, matches[i], width, MATCHES_GHOST_SIZE);
}

void fillup_matches(u8 *left_edges, u8 *right_edges, int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                matches[i][IGX(x, y, width, MATCHES_GHOST_SIZE)] =
                    left_edges[IGX(x, y, width, 30)] == right_edges[IGX(x+i, y, width, 30)];
                // ^ the +i accomplishes the sliding process
            }
        }
    }
}



// step 3

// a WxH size array used to keep scores
i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        scores[i] = ALLOCATE(i32, width * height);
}

void write_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image(scores[i], width, height, 0, IMTYPE_GRAY_INT, "score_edges", i);
}

void free_scores()
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        free(scores[i]);
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
                    int cur = IDX(x, y, width);
                    int rel = IGX(x + sx - half,
                                  y + sy - half,
                                  width, MATCHES_GHOST_SIZE);
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
        write_image(sum, width, height, 0, IMTYPE_GRAY_INT, "score_all", i);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = IDX(x, y, width);
                // record a score whenever there was a match-up
                if (matches[i][IGX(x, y, width, MATCHES_GHOST_SIZE)] == 1)
                    scores[i][index] = sum[index];
            }
        }
        write_image(scores[i], width, height, 0, IMTYPE_GRAY_INT, "score_edges", i);
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
                int index = IDX(x, y, width);
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
                int index = IDX(x, y, width);
                if (scores[i][index] == best_scores[index])
                    winning_shifts[index] = i+1;
            }
        }
    }
}



// step 4

i32 *fill_web_holes(i32 *web, int width, int height, int times)
{
    // each time though the loop, every pixel not on the web (i.e., every pixel that is not
    // zero to begin with) takes on the average elevation of its four neighbors. therefore,
    // the web pixels gradually "spread" their elevations across the holes, while they
    // themselves remain unchanged.
    i32 *tmp = ALLOCATE(i32, width * height);
    memcpy(tmp, web, sizeof(web[0]) * width * height);
    for (int i = 0; i < times; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (tmp[IDX(x, y, width)] == 0) {
                    web[IDX(x, y, width)] =
                        (tmp[IDX(x+1, y,   width)]
                       + tmp[IDX(x,   y+1, width)]
                       + tmp[IDX(x-1, y,   width)]
                       + tmp[IDX(x,   y-1, width)]) / 4;
                }
            }
        }
        SWAP(web, tmp, i32 *);
    }
    free(tmp);
    return web;
}

i32 image_max(i32 *im, int width, int height) { return array_max(im, width*height); }
i32 image_min(i32 *im, int width, int height) { return array_min(im, width*height); }

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
            image_output[IDX(x, y, width)] =
                ((web[IDX(x, y, width)] - min_elevation) % interval) == 0;
        }
    }
}



// main functions

typedef struct AlgorithmParams {
    double threshold;
    int square_width;
    int times;
    int lines_to_draw;
} AlgorithmParams;

void algorithm(double *first, double *second, int width, int height, AlgorithmParams params)
{
    u8 *first_edges  = ghost_alloc_u8(width, height, 30, 0),
       *second_edges = ghost_alloc_u8(width, height, 30, 0);
    i32 *buf = ALLOCATE(i32, width * height),
        *web = ALLOCATE(i32, width * height);
    u8 *out = ALLOCATE(u8, width * height);
    allocate_matches(width, height);
    allocate_scores(width, height);

    // first step: find edges in both images
    find_all_edges(first,  width, height, params.threshold, first_edges);
    find_all_edges(second, width, height, params.threshold, second_edges);
    write_image(first_edges,  width, height, 30, IMTYPE_BINARY, "edges", 1);
    write_image(second_edges, width, height, 30, IMTYPE_BINARY, "edges", 2);

    // second step: match edges between images
    fillup_matches(first_edges, second_edges, width, height);
    write_matches(width, height);

    // third step: compute scores for each pixel
    fillup_scores(width, height, params.square_width, buf);
    write_scores(width, height);
    find_highest_scoring_shifts(buf, web, width, height);
    write_image(buf, width, height, 0, IMTYPE_GRAY_INT, "score_best", 0);
    write_image(web, width, height, 0, IMTYPE_GRAY_INT, "web", 1);

    // fourth step: draw contour lines
    web = fill_web_holes(web, width, height, params.times);
    write_image(web, width, height, 0, IMTYPE_GRAY_INT, "web", 2);
    draw_contour_map(web, width, height, params.lines_to_draw, out);
    write_image(out, width, height, 0, IMTYPE_BINARY, "output", 0);

    GHOST_FREE(u8, first_edges,  width, 30);
    GHOST_FREE(u8, second_edges, width, 30);
    free(buf);
    free(web);
    free(out);
    free_matches(width);
    free_scores();
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

    AlgorithmParams params = {
        .threshold     = DEFAULT_THRESHOLD,
        .square_width  = DEFAULT_SQUARE_WIDTH,
        .times         = DEFAULT_TIMES,
        .lines_to_draw = DEFAULT_LINES
    };

    if (argc >= 4 && parse_double(argv[3], &params.threshold)) {
        fprintf(stderr, "error: threshold must be a number\n");
        return 1;
    }
    if (argc >= 5 && parse_int(argv[4], &params.square_width)) {
        fprintf(stderr, "error: square_width must be a number\n");
        return 1;
    }
    if (argc >= 6 && parse_int(argv[5], &params.times)) {
        fprintf(stderr, "error: times must be a number\n");
        return 1;
    }
    if (argc >= 7 && parse_int(argv[6], &params.lines_to_draw)) {
        fprintf(stderr, "error: lines must be a number\n");
        return 1;
    }

    double *first_ghost  = ghost_add_double(first.data,  first.width,  first.height,  1, 128.0);
    double *second_ghost = ghost_add_double(second.data, second.width, second.height, 1, 128.0);

    algorithm(first_ghost, second_ghost, first.width, first.height, params);

    GHOST_FREE(double, first_ghost,  first.width, 1);
    GHOST_FREE(double, second_ghost, first.width, 1);
    free(first.data);
    free(second.data);
    return 0;
}
