#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 16364
#define HEIGHT 16364

__global__ void update_kernel(unsigned int *grid, unsigned int *new_grid, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate in the grid
    int y = blockIdx.y; // y-coordinate in the grid

    if (x < width && y < height) {
        int bit_position = x % 32; // Determine the bit position in the 32-bit word
        int cell_block_index = x / 32; // Determine the 32-bit word (cell block) index

        unsigned int mask = 1u << bit_position;
        unsigned int cell = (grid[y * (width / 32) + cell_block_index]) & mask;

        // Count the neighbours
        int neighbour_count = 0;
        for (int displacement_y = -1; displacement_y <= 1; displacement_y++) {
            for (int displacement_x = -1; displacement_x <= 1; displacement_x++) {
                if (displacement_x == 0 && displacement_y == 0) continue; // Skip the cell itself

                int neighbour_x = x + displacement_x;
                int neighbour_y = y + displacement_y;

                // Check for boundary conditions
                if (neighbour_x >= 0 && neighbour_x < width && neighbour_y >= 0 && neighbour_y < height) {
                    int neighbour_bit_position = neighbour_x % 32;
                    int neighbour_block_index = neighbour_x / 32;
                    unsigned int neighbourMask = 1u << neighbour_bit_position;
                    unsigned int neighbour_cell = (grid[neighbour_y * (width / 32) + neighbour_block_index]) & neighbourMask;

                    if (neighbour_cell) neighbour_count++;
                }
            }
        }

        // Apply Game of Life rules
        if ((cell && (neighbour_count == 2 || neighbour_count == 3)) || (!cell && neighbour_count == 3)) {
            atomicOr(&new_grid[y * (width / 32) + cell_block_index], mask);
        } else {
            atomicAnd(&new_grid[y * (width / 32) + cell_block_index], ~mask);
        }
    }
}



// Function to initialize the grid with a bit-per-cell arrangement
void initialize_grid(unsigned int *grid) {
    // Seed for random number generation
    srand(time(NULL));

    // Iterate over each 32-bit integer in the grid
    for (int i = 0; i < HEIGHT * (WIDTH / 32); i++) {
        grid[i] = 0;
        for (int j = 0; j < 32; j++) {
            // Randomly set each bit to 0 or 1
            grid[i] |= (rand() % 2) << j;
        }
    }
}



int main() {
    size_t size = (WIDTH / 32) * HEIGHT * sizeof(unsigned int);
    unsigned int *grid, *new_grid;
    unsigned int *d_grid, *d_new_grid;
    cudaEvent_t start, end;
    float time = 0;

    // Allocate host memory
    grid = (unsigned int *)malloc(size);
    new_grid = (unsigned int *)malloc(size);

    // Initialize grid
    initialize_grid(grid);

    // Allocate device memory
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);

    // Copy initial grid to device
    cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(32, 1); // Each block contains 32 threads
    dim3 dimGrid(WIDTH / 32, HEIGHT);

    cudaEventCreate(&start);
	  cudaEventCreate(&end);
	  cudaEventRecord(start);
    // Main program loop
    for (int iter = 0; iter < 10; iter++) {
        update_kernel<<<dimGrid, dimBlock>>>(d_grid, d_new_grid, WIDTH, HEIGHT);

        // Swap grids
        unsigned int *temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;

        //printf("In iteration: %d\n", iter);
    }

    cudaEventRecord(end);
	  cudaEventSynchronize(end);

	  cudaEventElapsedTime(&time, start, end);

	  printf("The time to complete iterations is: %f\n", time);

    // Cleanup
    cudaFree(d_grid);
    cudaFree(d_new_grid);
    free(grid);
    free(new_grid);

    return 0;
}

