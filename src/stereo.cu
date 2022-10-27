#include "util.h"
#include <math.h>
#include "image.h"

#define NUM_SHIFTS 30
#define DEFAULT_THRESHOLD 0.15
#define DEFAULT_SQUARE_WIDTH 5
#define DEFAULT_TIMES 32
#define DEFAULT_LINES 10

// assume maximum thread no is 1024 (32*32)
#define BLOCK_DIM 32



// step 1

int __device__ find_edges_left_right(double *brightness, int width, int x, int y, double threshold)
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

int __device__ find_edges_top_bottom(double *brightness, int width, int x, int y, double threshold)
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

int __device__ find_edges_upleft_downright(double *brightness, int width, int x, int y, double threshold)
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

int __device__ find_edges_downleft_upright(double *brightness, int width, int x, int y, double threshold)
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

void __global__ find_all_edges(double *brightness, u8 *edges, int width, int height, double threshold)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    edges[IDX(x, y, width)] =
        find_edges_left_right(brightness, width, x, y, threshold)
     || find_edges_top_bottom(brightness, width, x, y, threshold)
     || find_edges_upleft_downright(brightness, width, x, y, threshold)
     || find_edges_downleft_upright(brightness, width, x, y, threshold);
}



// step 2

// a WxH size array used to keep matches
u8 __device__ *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    void *tmp[NUM_SHIFTS];
    for (int i = 0; i < NUM_SHIFTS; i++)
        cudaMalloc(&tmp[i], width * height * sizeof(matches[0]));
    cudaMemcpyToSymbol(matches, tmp, sizeof(tmp));
}

void write_matches(int width, int height)
{
    u8 *tmp[NUM_SHIFTS];
    cudaMemcpyFromSymbol(tmp, matches, sizeof(tmp));
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image_from_gpu(tmp[i], width, height, IMTYPE_BINARY, "matches", i);
}

void __global__ fillup_matches(u8 *left_edges, u8 *right_edges, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int index = idx(x,   y, width),
            shift = idx(x+i, y, width);
        // ^ the +i accomplishes the sliding process
        matches[i][index] = left_edges[index] == right_edges[shift];
    }
}

/*
// a WxH size array used to keep scores
i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    for (int i = 0; i < NUM_SHIFTS; i++)
        scores[i] = (i32 *) xmalloc(width * height, sizeof(scores[0]));
}

 * for each pixel in parallel:
 *
 * *.. .*. ..* ... ... ... ... ... ...
 * ... ... ... *.. .*. ..* ... ... ...
 * ... ... ... ... ... ... *.. .*. ..*
 *
 * (where the considered pixel is at the center and square_width = 3)
 * pixels must be a binary image.
void addup_pixels_in_square(u8 *pixels, int width, int height, int square_width, i32 *total)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int half = square_width / 2;
    memset(total, 0, sizeof(total[0]) * width * height);
    for (int sy = 0; sy < square_width; sy++) {
        for (int sx = 0; sx < square_width; sx++) {
            total[idx(x, y, width)] +=
                (i32) pixels[idx(x + sx - half,
                                 y + sy - half, width)];
        }
    }
}

void fillup_scores(int width, int height, int square_width, i32 *sum)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < NUM_SHIFTS; i++) {
        addup_pixels_in_square(matches[i], width, height, square_width, sum);
        int index = idx(x, y, width);
        // record a score whenever there was a match-up
        if (matches[i][index] == 1)
            scores[i][index] = sum[index];
    }
}

// this function computes the web of known shifts. recall that
// the shift at each pixel corresponds directly to the elevation.
void find_highest_scoring_shifts(i32 *best_scores, i32 *winning_shifts, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    memset(best_scores,    0, sizeof(best_scores[0])    * width * height);
    memset(winning_shifts, 0, sizeof(winning_shifts[0]) * width * height);
    // the following loop makes sure that each pixel in the best_scores
    // image contains the maximum score found at any shift.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int index = idx(x, y, width);
        if (scores[i][index] > best_scores[index])
            best_scores[index] = scores[i][index];
    }
    write_image(best_scores, width, height, IMTYPE_GRAY_INT, "score_best", 0);
    // the following loop records a 'winning' shift at every pixel
    // whose score is the best.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int index = idx(x, y, width);
        if (scores[i][index] == best_scores[index])
            winning_shifts[index] = i+1;
    }
}

void __global__ fill_web_holes(i32 *web, i32 *tmp, int width, int times)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < times; i++) {
        if (web[idx(x, y, width)] == 0) {
            web[idx(x, y, width)] =
                (tmp[idx(x+1, y,   width)]
               + tmp[idx(x,   y+1, width)]
               + tmp[idx(x-1, y,   width)]
               + tmp[idx(x,   y-1, width)]) / 4;
        }
    }
}

i32 *fill_web_holes(i32 *web, int width, int height, int times)
{
    // each time though the loop, every pixel not on the web (i.e., every pixel that is not
    // zero to begin with) takes on the average elevation of its four neighbors. therefore,
    // the web pixels gradually "spread" their elevations across the holes, while they
    // themselves remain unchanged.
    i32 *tmp; cudaMalloc(&tmp, width * height, sizeof(web[0]));
    cudaMemcpy(tmp, web, sizeof(web[0]) * width * height, cudaMemcpyDeviceToDevice);
    for (int i = 0; i < times; i++) {
        fill_web_holes<<<>>>(web, tmp, width, times);
        SWAP(web, tmp, i32 *);
    }
    cudaFree(tmp);
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

*/



typedef struct AlgorithmParams {
    double threshold;
    int square_width;
    int times;
    int lines_to_draw;
} AlgorithmParams;

void algorithm(double *first, double *second, int width, int height, AlgorithmParams params)
{
    const int num_blocks_side = ceil_div(width, BLOCK_DIM);
    const dim3 num_blocks = dim3(num_blocks_side, num_blocks_side);
    const dim3 block_dim  = dim3(BLOCK_DIM, BLOCK_DIM);

    // first step: find edges in both images
    u8 *first_edges; cudaMalloc(&first_edges, width * height * sizeof(u8));
    find_all_edges<<<num_blocks, block_dim>>>(first, first_edges, width, height, params.threshold);
    write_image_from_gpu(first_edges, width, height, IMTYPE_BINARY, "edges", 1);

    u8 *second_edges; cudaMalloc(&second_edges, width * height * sizeof(u8));
    find_all_edges<<<num_blocks, block_dim>>>(second, second_edges, width, height, params.threshold);
    write_image_from_gpu(second_edges, width, height, IMTYPE_BINARY, "edges", 2);

    // second step: match edges between images
    allocate_matches(width, height);
    fillup_matches<<<num_blocks, block_dim>>>(first_edges, second_edges, width, height);
    write_matches(width, height);

    /*
    // third step: compute scores for each pixel
    i32 *buf            = (i32 *) xmalloc(width * height, sizeof(i32)),
        *winning_shifts = (i32 *) xmalloc(width * height, sizeof(i32));
    allocate_scores(width, height);
    fillup_scores(width, height, 5, buf);
    find_highest_scoring_shifts(buf, winning_shifts, width, height);
    write_image_from_gpu(winning_shifts, width, height, IMTYPE_GRAY_INT, "web", 1);

    // fourth step: draw contour lines
    i32 *web = winning_shifts;
    u8 *out = (u8 *) xmalloc(width * height, sizeof(u8));
    web = fill_web_holes(web, width, height, times);
    write_image_from_gpu(web, width, height, IMTYPE_GRAY_INT, "web", 2);
    draw_contour_map(web, width, height, lines_to_draw, out);
    write_image_from_gpu(out, width, height, IMTYPE_BINARY, "output", 0);
    */
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

    double *first_img;
    cudaMalloc(&first_img,  first.width  * first.height  * sizeof(double));
    cudaMemcpy(first_img, first.data, first.width * first.height * sizeof(double), cudaMemcpyHostToDevice);

    double *second_img;
    cudaMalloc(&second_img, second.width * second.height * sizeof(double));
    cudaMemcpy(second_img, second.data, second.width * second.height * sizeof(double), cudaMemcpyHostToDevice);

    algorithm(first_img, second_img, first.width, first.height, params);

    cudaDeviceSynchronize();

    return 0;
}
