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

__device__ int find_edges_left_right(double *brightness, int width, int x, int y, double threshold)
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

__device__ int find_edges_top_bottom(double *brightness, int width, int x, int y, double threshold)
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

__device__ int find_edges_upleft_downright(double *brightness, int width, int x, int y, double threshold)
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

__device__ int find_edges_downleft_upright(double *brightness, int width, int x, int y, double threshold)
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

__global__ void find_all_edges(double *brightness, int width, int height, double threshold, u8 *edges)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    edges[IGX(x, y, width, 30)] =
        find_edges_left_right(brightness, width, x, y, threshold)
     || find_edges_top_bottom(brightness, width, x, y, threshold)
     || find_edges_upleft_downright(brightness, width, x, y, threshold)
     || find_edges_downleft_upright(brightness, width, x, y, threshold);
}



// step 2

// a WxH size array used to keep matches
__device__ u8 *matches[NUM_SHIFTS];

void allocate_matches(int width, int height)
{
    u8 *tmp[NUM_SHIFTS];
    for (int i = 0; i < NUM_SHIFTS; i++)
        tmp[i] = ghost_alloc_gpu_u8(width, height, MATCHES_GHOST_SIZE, 0);
    checkCudaErrors(cudaMemcpyToSymbol(matches, tmp, sizeof(tmp)));
}

void write_matches(int width, int height)
{
    u8 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, matches, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image_from_gpu(tmp[i], width, height, MATCHES_GHOST_SIZE, IMTYPE_BINARY, "matches", i);
}

void free_matches(int width)
{
    u8 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, matches, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        GHOST_FREE_GPU(u8, tmp[i], width, MATCHES_GHOST_SIZE);
}

__global__ void fillup_matches(u8 *left_edges, u8 *right_edges, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < NUM_SHIFTS; i++)
        matches[i][IGX(x, y, width, MATCHES_GHOST_SIZE)] =
            left_edges[IGX(x, y, width, 30)] == right_edges[IGX(x+i, y, width, 30)];
        // ^ the +i accomplishes the sliding process
}



// step 3

// a WxH size array used to keep scores
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
    i32 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, scores, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        write_image_from_gpu(tmp[i], width, height, 0, IMTYPE_GRAY_INT, "score_edges", i);
}

void free_scores()
{
    i32 *tmp[NUM_SHIFTS];
    checkCudaErrors(cudaMemcpyFromSymbol(tmp, scores, sizeof(tmp)));
    for (int i = 0; i < NUM_SHIFTS; i++)
        checkCudaErrors(cudaFree(tmp[i]));
}

__device__ void addup_pixels_in_square(u8 *pixels, int width, int height, int square_width, i32 *total)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int half = square_width / 2;
    memset(total, 0, sizeof(total[0]) * width * height);
    for (int sy = 0; sy < square_width; sy++) {
        for (int sx = 0; sx < square_width; sx++) {
            int cur = IDX(x, y, width);
            int rel = IGX(x + sx - half,
                          y + sy - half,
                          width, MATCHES_GHOST_SIZE);
            total[cur] += (i32) pixels[rel];
        }
    }
}

__global__ void fillup_scores(int width, int height, int square_width, i32 *sum)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    for (int i = 0; i < NUM_SHIFTS; i++) {
        addup_pixels_in_square(matches[i], width, height, square_width, sum);
        // record a score whenever there was a match-up
        int index = IDX(x, y, width);
        if (matches[i][IGX(x, y, width, MATCHES_GHOST_SIZE)] == 1)
            scores[i][index] = sum[index];
    }
}

__global__ void find_highest_scoring_shifts(i32 *best_scores, i32 *winning_shifts, int width, int height)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    memset(best_scores,    0, sizeof(best_scores[0])    * width * height);
    memset(winning_shifts, 0, sizeof(winning_shifts[0]) * width * height);
    // the following loop makes sure that each pixel in the best_scores
    // image contains the maximum score found at any shift.
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int index = IDX(x, y, width);
        if (scores[i][index] > best_scores[index])
            best_scores[index] = scores[i][index];
    }
    for (int i = 0; i < NUM_SHIFTS; i++) {
        int index = IDX(x, y, width);
        if (scores[i][index] == best_scores[index])
            winning_shifts[index] = i+1;
    }
}



typedef struct AlgorithmParams {
    double threshold;
    int square_width;
    int times;
    int lines_to_draw;
} AlgorithmParams;

void algorithm(double *first, double *second, int width, int height, AlgorithmParams params)
{
    const int num_blocks_side = ceil_div(width, BLOCK_DIM_2D);
    const dim3 num_blocks = dim3(num_blocks_side, num_blocks_side);
    const dim3 block_dim  = dim3(BLOCK_DIM_2D, BLOCK_DIM_2D);

    u8 *first_edges  = ghost_alloc_gpu_u8(width, height, 30, 0),
       *second_edges = ghost_alloc_gpu_u8(width, height, 30, 0);
    i32 *buf = ALLOCATE_GPU(i32, width * height),
        *web = ALLOCATE_GPU(i32, width * height);
    u8 *out = ALLOCATE_GPU(u8, width * height);
    allocate_matches(width, height);
    allocate_scores(width, height);

    // first step: find edges in both images
    find_all_edges<<<num_blocks, block_dim>>>(first,  width, height, params.threshold, first_edges);
    find_all_edges<<<num_blocks, block_dim>>>(second, width, height, params.threshold, second_edges);
    write_image_from_gpu(first_edges,  width, height, 30, IMTYPE_BINARY, "edges", 1);
    write_image_from_gpu(second_edges, width, height, 30, IMTYPE_BINARY, "edges", 2);

    // second step: match edges between images
    fillup_matches<<<num_blocks, block_dim>>>(first_edges, second_edges, width, height);
    write_matches(width, height);

    // third step: compute scores for each pixel
    fillup_scores<<<num_blocks, block_dim>>>(width, height, params.square_width, buf);
    write_scores(width, height);
    find_highest_scoring_shifts<<<num_blocks, block_dim>>>(buf, web, width, height);
    write_image_from_gpu(buf, width, height, 0, IMTYPE_GRAY_INT, "score_best", 0);
    write_image_from_gpu(web, width, height, 0, IMTYPE_GRAY_INT, "web", 1);

    GHOST_FREE_GPU(u8, first_edges,  width, 30);
    GHOST_FREE_GPU(u8, second_edges, width, 30);
    free_matches(width);
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

    double *first_ghost  = ghost_add_gpu_double(first.data,  first.width, first.height, 1, 128.0, cudaMemcpyHostToDevice);
    double *second_ghost = ghost_add_gpu_double(second.data, first.width, first.height, 1, 128.0, cudaMemcpyHostToDevice);

    algorithm(first_ghost, second_ghost, first.width, first.height, params);
    cudaDeviceSynchronize();

    GHOST_FREE_GPU(double, first_ghost,  first.width, 1);
    GHOST_FREE_GPU(double, second_ghost, first.width, 1);
    free(first.data);
    free(second.data);

    return 0;
}
