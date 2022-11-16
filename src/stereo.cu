#include "util.h"
#include <math.h>
#include "image.h"
#include "ghost.h"

#define NUM_SHIFTS 30
#define DEFAULT_THRESHOLD 0.15
#define DEFAULT_SQUARE_WIDTH 5
#define DEFAULT_TIMES 32
#define DEFAULT_LINES 10

#define DECLARE_BLOCKS(w, h) \
    const int BLOCKS_WIDTH = ceil_div(width, BLOCK_DIM_SIDE);   \
    const int BLOCKS_HEIGHT = ceil_div(height, BLOCK_DIM_SIDE); \
    const dim3 NUM_BLOCKS = dim3(BLOCKS_WIDTH, BLOCKS_HEIGHT);

#define DECLARE_INDEXES(w, h) \
    int x = threadIdx.x + blockIdx.x * blockDim.x;  \
    int y = threadIdx.y + blockIdx.y * blockDim.y;  \
    if (x < (w) || y < (h))                         \
        return;



// step 1

__device__ int find_edges_left_right(double *brightness,
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

__device__ int find_edges_top_bottom(double *brightness,
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

__device__ int find_edges_upleft_downright(double *brightness,
        int height, int width, int x, int y, double threshold)
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

__device__ int find_edges_downleft_upright(double *brightness,
        int height, int width, int x, int y, double threshold)
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

__global__ void find_all_edges(double *brightness, int w, int h,
        double threshold, u8 *edges)
{
    DECLARE_INDEXES(w, h)
    edges[IDX(x, y, w)] =
              find_edges_left_right(brightness, w, h, x, y, threshold)
     ||       find_edges_top_bottom(brightness, w, h, x, y, threshold)
     || find_edges_upleft_downright(brightness, w, h, x, y, threshold)
     || find_edges_downleft_upright(brightness, w, h, x, y, threshold);
}



// step 2

__device__ u8 *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    u8 *tmp[NUM_SHIFTS];
    for (int i = 0; i < NUM_SHIFTS; i++)
        tmp[i] = ALLOCATE_GPU(u8, width * height);
    checkCudaErrors(cudaMemcpyToSymbol(matches, tmp, sizeof(tmp)));
}

void write_matches(int width, int height)
{
#ifndef NO_WRITES
    u8 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, matches, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_gpu_image(tmp[i], width, height, 0, IMTYPE_BINARY, make_filename("matches", PAR, i));
#endif
}

void free_matches()
{
    u8 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, matches, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        checkCudaErrors(cudaFree(tmp[i]));
}

// this function records the edge-pixel match-ups at every shift
__global__ void fillup_matches(u8 *left_edges, u8 *right_edges,
        int width, int height)
{
    DECLARE_INDEXES(width, height)
    int index = IDX(x, y, width);
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int shift = idx(x+i, y, width, height);
        // ^ the +i accomplishes the sliding process
        matches[i][index] = left_edges[index] == right_edges[shift];
    }
}



// step 3

// the square for each pixel is to be centered on that pixel.
// the double for loop is slightly different than the original,
// going from -half to +half.
__global__ void addup_pixels_in_square(int i, int width, int height,
        int square_width, i32 *total)
{
    DECLARE_INDEXES(width, height)
    u8 *pixels = matches[i];
    int cur = IDX(x, y, width);
    int half = square_width / 2;
    for (int sy = -half; sy <= half; sy++) {
        for (int sx = -half; sx <= half; sx++) {
            int rel = idx(x + sx, y + sy, width, height);
            total[cur] += (i32) pixels[rel];
        }
    }
}

__device__ i32 *scores[NUM_SHIFTS];

void allocate_scores(int width, int height)
{
    i32 *tmp[NUM_SHIFTS];
    for (int i = 0; i < NUM_SHIFTS; i++)
        tmp[i] = ALLOCATE_GPU(i32, width * height);
    checkCudaErrors(cudaMemcpyToSymbol(scores, tmp, sizeof(tmp)));
}

void write_scores(int width, int height)
{
#ifndef NO_WRITES
    i32 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, scores, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_gpu_image(tmp[i], width, height, 0, IMTYPE_GRAY_INT, make_filename("scores", PAR, i));
#endif
}

void free_scores()
{
    i32 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, scores, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        checkCudaErrors(cudaFree(tmp[i]));
}

__global__ void record_score(int i, i32 *sum, int width, int height)
{
    DECLARE_INDEXES(width, height)
    int index = IDX(x, y, width);
    // record a score whenever there was a match-up
    if (matches[i][index] == 1)
        scores[i][index] = sum[index];
}

void fillup_scores(int width, int height, int square_width, i32 *sum)
{
    DECLARE_BLOCKS(width, height)
    for (int i = 0; i < NUM_SHIFTS; i++) {
        cudaMemset(sum, 0, sizeof(sum[0]) * width * height);
        addup_pixels_in_square<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(
            i, width, height, square_width, sum
        );
        write_gpu_image(sum, width, height, 0, IMTYPE_GRAY_INT, make_filename("score_all", PAR, i));
        record_score<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(
            i, sum, width, height
        );
    }
}

// this function computes the web of known shifts. recall that
// the shift at each pixel corresponds directly to the elevation.
__global__ void find_highest_scoring_shifts(i32 *best_scores,
        i32 *winning_shifts, int width, int height)
{
    DECLARE_INDEXES(width, height)
    int index = IDX(x, y, width);
    // the following for loop makes sure that each pixel in the
    // 'best_scores' variable contains the maximum score found at any shift.
    for (int i = 0; i < NUM_SHIFTS; i++)
        best_scores[index] = MAX(scores[i][index], best_scores[index]
    // the following for loop records a "winning"
    // shift at every pixel whose score is the best.
    for (int i = 0; i < NUM_SHIFTS; i++)
        if (scores[i][index] == best_scores[index])
            winning_shifts[index] = i+1;
}



// step 4

__global__ void fill_web_holes_step(i32 *web, i32 *tmp, int width)
{
    DECLARE_INDEXES(width, height)
    if (tmp[IDX(x, y, width)] == 0)
        web[IDX(x, y, width)] =
            (tmp[IDX(x+1, y,   width)]  // neighbor to the right
           + tmp[IDX(x,   y+1, width)]  // neighbor above
           + tmp[IDX(x-1, y,   width)]  // neighbor to the left
           + tmp[IDX(x,   y-1, width)]) // neighbor below
           / 4;
}

// each time though the loop, every pixel not on the web (i.e., every pixel that is not
// zero to begin with) takes on the average elevation of its four neighbors. therefore,
// the web pixels gradually "spread" their elevations across the holes, while they
// themselves remain unchanged.
i32 *fill_web_holes(i32 *web, i32 *tmp, int width, int height, int times)
{
    DECLARE_BLOCKS(width, height)
    for (int i = 0; i < times; i++) {
        fill_web_holes_step<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(web, tmp, width);
        SWAP(i32 *, web, tmp);
    }
    // this is now a more or less smooth surface.
    return web;
}

i32 image_max(i32 *im, int width, int height) { return array_max_gpu(im, width*height); }
i32 image_min(i32 *im, int width, int height) { return array_min_gpu(im, width*height); }

__global__ void draw_contour_map_kernel(i32 *web, int width,
        int num_lines, i32 max_elevation, i32 min_elevation, u8 *out)
{
    // the idea is to divide the whole range of elevations into a number
    // of intervals, then to draw a contour line at every interval.
    DECLARE_INDEXES(width, height)
    int i = IDX(x, y, width);
    // the variable 'interval' tells us how many
    // elevations, or shifts, to skip between contour lines.
    i32 range    = max_elevation - min_elevation,
        interval = range / num_lines;
    // this line draws all the elevation contours at once.
    out[i] = ((web[i] - min_elevation) % interval) == 0;
}

void draw_contour_map(i32 *web, int width, int height,
        int num_lines, u8 *out)
{
    DECLARE_BLOCKS(width, height)
    i32 immax = image_max(web, width, height),
        immin = image_min(web, width, height);
    draw_contour_map_kernel<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(
        web, width, num_lines, immax, immin, out
    );
}



typedef struct AlgorithmParams {
    double threshold;
    int square_width;
    int times;
    int lines_to_draw;
} AlgorithmParams;

void algorithm(double *first, double *second, int width, int height, AlgorithmParams params)
{
    DECLARE_BLOCKS(width, height)
    u8 *first_edges  = ALLOCATE_GPU(u8, width * height),
       *second_edges = ALLOCATE_GPU(u8, width * height);
    i32 *buf         = ALLOCATE_GPU(i32, width * height);
    i32 *web         = ALLOCATE_GPU(i32, width * height);
    i32 *tmp         = ALLOCATE_GPU(i32, width * height);
    u8 *out          = ALLOCATE_GPU(u8, width * height);
    allocate_matches(width, height);
    allocate_scores(width, height);

    double t1 = get_time();

    // first step: find edges in both images
    find_all_edges<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(first,  width, height, params.threshold, first_edges);
    find_all_edges<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(second, width, height, params.threshold, second_edges);
    write_gpu_image(first_edges,  width, height, 0, IMTYPE_BINARY, make_filename("edges", PAR, 1));
    write_gpu_image(second_edges, width, height, 0, IMTYPE_BINARY, make_filename("edges", PAR, 2));

    // second step: match edges between images
    fillup_matches<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(first_edges, second_edges, width, height);
    write_matches(width, height);

    fillup_scores(width, height, params.square_width, buf);
    write_scores(width, height);
    cudaMemset(buf, 0, sizeof(buf[0]) * width * height);
    find_highest_scoring_shifts<<<NUM_BLOCKS, BLOCK_DIM_2D>>>(buf, web, width, height);
    write_gpu_image(buf, width, height, 0, IMTYPE_GRAY_INT, make_filename("score_best", PAR, 0));
    write_gpu_image(web, width, height, 0, IMTYPE_GRAY_INT, make_filename("web", PAR, 1));

    // third step: draw contour lines
    web = fill_web_holes(web, tmp, width, height, params.times);
    checkCudaErrors(cudaMemcpy(tmp, web, sizeof(web[0]) * width * height, cudaMemcpyDeviceToDevice));
    write_gpu_image(web, width, height, 0, IMTYPE_GRAY_INT, make_filename("web", PAR, 2));
    draw_contour_map(web, width, height, params.lines_to_draw, out);
    write_gpu_image(out, width, height, 0, IMTYPE_BINARY, make_filename("output", PAR, 0));

    cudaDeviceSynchronize();
    double t2 = get_time();
    double elapsed = t2 - t1;
    printf("width = %d, height = %d, t1 = %f, t2 = %f, elapsed = %f\n", width, height, t1, t2, elapsed);

    checkCudaErrors(cudaFree(first_edges));
    checkCudaErrors(cudaFree(second_edges));
    checkCudaErrors(cudaFree(web));
    checkCudaErrors(cudaFree(out));
    checkCudaErrors(cudaFree(buf));
    checkCudaErrors(cudaFree(tmp));
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

    double *first_gpu  = MAKE_GPU_COPY(double, first.data,  first.width * first.height),
           *second_gpu = MAKE_GPU_COPY(double, second.data, first.width * first.height);
    algorithm(first_gpu, second_gpu, first.width, first.height, params);

    free(first.data);
    free(second.data);
    return 0;
}

