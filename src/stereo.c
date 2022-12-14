#include "util.h"
#include <math.h>
#include "image.h"
#include "ghost.h"

#define NUM_SHIFTS 30
#define DEFAULT_THRESHOLD 0.15
#define DEFAULT_SQUARE_WIDTH 21
#define DEFAULT_TIMES 32
#define DEFAULT_LINES 10



// step 1

int find_edges_left_right(double *brightness,
        int width, int height, int x, int y, double threshold)
{
    double avg_left  = (brightness[idx(x-1, y-1, width, height)]
                     +  brightness[idx(x-1, y  , width, height)]
                     +  brightness[idx(x-1, y+1, width, height)]) / 3.0;
    double avg_right = (brightness[idx(x+1, y-1, width, height)]
                     +  brightness[idx(x+1, y  , width, height)]
                     +  brightness[idx(x+1, y+1, width, height)]) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall,
                                                0.0, 1.0);
}

int find_edges_top_bottom(double *brightness,
        int width, int height, int x, int y, double threshold)
{
    double avg_left  = (brightness[idx(x-1, y-1, width, height)]
                     +  brightness[idx(x  , y-1, width, height)]
                     +  brightness[idx(x+1, y-1, width, height)]) / 3.0;
    double avg_right = (brightness[idx(x-1, y+1, width, height)]
                     +  brightness[idx(x  , y+1, width, height)]
                     +  brightness[idx(x+1, y+1, width, height)]) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall,
                                                0.0, 1.0);
}

int find_edges_upleft_downright(double *brightness,
        int width, int height, int x, int y, double threshold)
{
    double avg_left  = (brightness[idx(x-1, y-1, width, height)]
                     +  brightness[idx(x  , y-1, width, height)]
                     +  brightness[idx(x-1, y  , width, height)]) / 3.0;
    double avg_right = (brightness[idx(x+1, y  , width, height)]
                     +  brightness[idx(x  , y+1, width, height)]
                     +  brightness[idx(x+1, y+1, width, height)]) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall,
                                                0.0, 1.0);
}

int find_edges_downleft_upright(double *brightness,
        int width, int height, int x, int y, double threshold)
{
    double avg_left  = (brightness[idx(x-1, y+1, width, height)]
                     +  brightness[idx(x  , y+1, width, height)]
                     +  brightness[idx(x-1, y  , width, height)]) / 3.0;
    double avg_right = (brightness[idx(x  , y-1, width, height)]
                     +  brightness[idx(x+1, y-1, width, height)]
                     +  brightness[idx(x+1, y  , width, height)]) / 3.0;
    double overall   = (avg_left + avg_right) / 2.0;
    return fabs(avg_left - avg_right) > CLAMP(threshold * overall,
                                                0.0, 1.0);
}

void find_all_edges(double *brightness, int w, int h,
        double threshold, u8 *edges)
{
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y++) {
            edges[IDX(x, y, w)] =
                      find_edges_left_right(brightness, w, h, x, y, threshold)
             ||       find_edges_top_bottom(brightness, w, h, x, y, threshold)
             || find_edges_upleft_downright(brightness, w, h, x, y, threshold)
             || find_edges_downleft_upright(brightness, w, h, x, y, threshold);
        }
    }
}



// step 2

u8 *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        matches[i] = ALLOCATE(u8, width * height);
}

void write_matches(int width, int height)
{
#ifndef NO_WRITES
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image(matches[i], width, height, 0, IMTYPE_BINARY, make_filename("matches", SER, i));
#endif
}

void free_matches()
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        free(matches[i]);
}

// this function records the edge-pixel match-ups at every shift
void fillup_matches(u8 *left_edges, u8 *right_edges,
        int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = IDX(x,   y, width),
                    shift = idx(x+i, y, width, height);
                // ^ the +i accomplishes the sliding process
                matches[i][index] =
                    left_edges[index] == right_edges[shift];
            }
        }
    }
}

// the square for each pixel is to be centered on that pixel.
// the double for loop is slightly different than the original,
// going from -half to +half.
void addup_pixels_in_square(u8 *pixels, int width, int height,
        int square_width, i32 *total)
{
    int half = square_width / 2;
    for (int sy = -half; sy <= half; sy++) {
        for (int sx = -half; sx <= half; sx++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int cur = IDX(x, y, width);
                    int rel = idx(x + sx, y + sy,
                                  width, height);
                    total[cur] += (i32) pixels[rel];
                }
            }
        }
    }
}

i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        scores[i] = ALLOCATE(i32, width * height);
}

void write_scores(int width, int height)
{
#ifndef NO_WRITES
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image(scores[i], width, height, 0, IMTYPE_GRAY_INT, make_filename("scores", SER, i));
#endif
}

void free_scores()
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        free(scores[i]);
}

void record_score(int i, i32 *sum, int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = IDX(x, y, width);
            // record a score whenever there was a match-up
            if (matches[i][index] == 1)
                scores[i][index] = sum[index];
        }
    }
}

void fillup_scores(int width, int height, int square_width, i32 *sum)
{
    for (int i = 0; i < NUM_SHIFTS; i++) {
        memset(sum, 0, sizeof(sum[0]) * width * height);
        addup_pixels_in_square(matches[i], width, height, square_width, sum);
        write_image(sum, width, height, 0, IMTYPE_GRAY_INT, make_filename("score_all", SER, i));
        record_score(i, sum, width, height);
    }
}

// this function computes the web of known shifts. recall that
// the shift at each pixel corresponds directly to the elevation.
void find_highest_scoring_shifts(i32 *best_scores,
            i32 *winning_shifts, int width, int height)
{
    // the following loop makes sure that each pixel in the
    // 'best_scores' image contains the maximum score found at any shift.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int index = IDX(x, y, width);
                best_scores[index] = MAX(scores[i][index], best_scores[index]);
            }
        }
    }
    // the following loop records a "winning" shift at every pixel
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



// step 3

// each time though the loop, every pixel not on the web (i.e., every pixel that is not
// zero to begin with) takes on the average elevation of its four neighbors. therefore,
// the web pixels gradually "spread" their elevations across the holes, while they
// themselves remain unchanged.
i32 *fill_web_holes(i32 *web, int width, int height, int times)
{
    i32 *tmp = ALLOCATE(i32, width * height);
    memcpy(tmp, web, sizeof(web[0]) * width * height);
    for (int i = 0; i < times; i++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (tmp[IDX(x, y, width)] == 0) {
                    web[IDX(x, y, width)] =
                        (tmp[IDX(x+1, y,   width)]  // neighbor to the right
                       + tmp[IDX(x,   y+1, width)]  // neighbor above
                       + tmp[IDX(x-1, y,   width)]  // neighbor to the left
                       + tmp[IDX(x,   y-1, width)]) // neighbor below
                       / 4;
                }
            }
        }
        SWAP(i32 *, web, tmp);
    }
    free(tmp);
    return web;
}

i32 image_max(i32 *im, int width, int height) { return array_max(im, width*height); }
i32 image_min(i32 *im, int width, int height) { return array_min(im, width*height); }

void draw_contour_map(i32 *web, int width, int height,
        int num_lines, u8 *out)
{
    i32 max_elevation = image_max(web, width, height),
        min_elevation = image_min(web, width, height);
    // the idea is to divide the whole range of elevations into a number
    // of intervals, then to draw a contour line at every interval.
    // the variable 'interval' tells us how many
    // elevations, or shifts, to skip between contour lines.
    i32 range    = max_elevation - min_elevation,
        interval = range / num_lines;
    // this loop draws all the elevation.
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = IDX(x, y, width);
            out[index] = ((web[index] - min_elevation) % interval) == 0;
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
    u8 *first_edges  = ALLOCATE(u8, width * height),
       *second_edges = ALLOCATE(u8, width * height);
    i32 *buf         = ALLOCATE(i32, width * height),
        *web         = ALLOCATE(i32, width * height);
    u8 *out          = ALLOCATE(u8, width * height);
    allocate_matches(width, height);
    allocate_scores(width, height);

    double t1 = get_time();

    // first step: find edges in both images
    find_all_edges(first,  width, height, params.threshold, first_edges);
    find_all_edges(second, width, height, params.threshold, second_edges);
    write_image(first_edges,  width, height, 0, IMTYPE_BINARY, make_filename("edges", SER, 1));
    write_image(second_edges, width, height, 0, IMTYPE_BINARY, make_filename("edges", SER, 2));

    // second step: match edges between images
    fillup_matches(first_edges, second_edges, width, height);
    write_matches(width, height);

    fillup_scores(width, height, params.square_width, buf);
    write_scores(width, height);
    memset(buf, 0, sizeof(buf[0]) * width * height);
    find_highest_scoring_shifts(buf, web, width, height);
    write_image(buf, width, height, 0, IMTYPE_GRAY_INT, make_filename("score_best", SER, 0));
    write_image(web, width, height, 0, IMTYPE_GRAY_INT, make_filename("web", SER, 1));

    // third step: draw contour lines
    web = fill_web_holes(web, width, height, params.times);
    write_image(web, width, height, 0, IMTYPE_GRAY_INT, make_filename("web", SER, 2));
    draw_contour_map(web, width, height, params.lines_to_draw, out);
    write_image(out, width, height, 0, IMTYPE_BINARY, make_filename("output", SER, 0));

    double t2 = get_time();
    double elapsed = t2 - t1;
    printf("width = %d, height = %d, t1 = %f, t2 = %f, elapsed = %f\n", width, height, t1, t2, elapsed);

    free(first_edges);
    free(second_edges);
    free(buf);
    free(web);
    free(out);
    free_matches();
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

    if (params.threshold < 0.0 || params.threshold > 1.0) {
        fprintf(stderr, "error: threshold must be between 0 and 1\n");
        return 1;
    }
    if (params.square_width > first.width || params.square_width > first.height) {
        fprintf(stderr, "error: square width must not be higher than image width/height\n");
        return 1;
    }

    algorithm(first.data, second.data, first.width, first.height, params);

    free(first.data);
    free(second.data);
    return 0;
}
