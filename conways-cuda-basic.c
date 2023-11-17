#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define WIDTH 16364
#define HEIGHT 16364

__global__ void update_kernel(int *grid, int *new_grid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < WIDTH && y < HEIGHT) {
        int live_neighbours = 0;

        // Count live neighbours
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                if (i == 0 && j == 0) continue;
                int col = (x + i + WIDTH) % WIDTH;
                int row = (y + j + HEIGHT) % HEIGHT;
                live_neighbours += grid[row * WIDTH + col];
            }
        }

        // Apply rules of the game
        int idx = y * WIDTH + x;
        if (grid[idx] == 1 && (live_neighbours < 2 || live_neighbours > 3))
            new_grid[idx] = 0;
        else if (grid[idx] == 0 && live_neighbours == 3)
            new_grid[idx] = 1;
        else
            new_grid[idx] = grid[idx];
    }
}

void initialize_grid(int *grid) {
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            grid[i * WIDTH + j] = rand() % 2;
        }
    }
}

int main() {
    size_t size = WIDTH * HEIGHT * sizeof(int);
    int *grid, *new_grid;
    int *d_grid, *d_new_grid;
    cudaEvent_t start, end;
    float time = 0;

    // Allocate host memory
    grid = (int *)malloc(size);
    new_grid = (int *)malloc(size);

    initialize_grid(grid);

    // Allocate device memory
    cudaMalloc(&d_grid, size);
    cudaMalloc(&d_new_grid, size);

    // Copy initial grid to device
    cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((WIDTH + dimBlock.x - 1) / dimBlock.x, (HEIGHT + dimBlock.y - 1) / dimBlock.y);
    
    cudaEventCreate(&start);
	  cudaEventCreate(&end);
	  cudaEventRecord(start);

    // Main loop
    for (int iter = 0; iter < 10; iter++) {
        // Update grid
        update_kernel<<<dimGrid, dimBlock>>>(d_grid, d_new_grid);

        // Swap grids
        int *temp = d_grid;
        d_grid = d_new_grid;
        d_new_grid = temp;
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
