# CUDA-Based-GPU-vs-GPU-Number-Guessing-Battle-Game
## Project Overview

This project demonstrates a simple GPU vs GPU Number Battle Game using CUDA. Two GPU players automatically generate random numbers using CUDA kernels. The generated numbers are compared, and the winner for each round is displayed in the terminal.

The project is designed to demonstrate CUDA programming concepts such as GPU computation, CUDA kernel execution, parallel processing, and device-to-host memory transfer through a simple gaming application. Multiple rounds are executed automatically, and the results are displayed after every round.

## Features
GPU vs GPU gameplay
CUDA kernel execution
Automatic number generation
Multiple round execution
Terminal-based output visualization
Winner determination after every round
Demonstration of parallel processing concepts
## Technologies Used
CUDA Toolkit
NVIDIA GPU
C++
Google Colab / CUDA Environment
CUDA Runtime API
## Project Structure
GPU-Number-Battle-CUDA/
│
├── gpu_number_battle.cu
├── README.md
├── output.txt
├── execution_log.txt
└── demo_video_link.txt

## Program
```
%%writefile gpu_number_battle.cu

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

__global__ void generateNumbers(int *numbers, int seed)
{
    int idx = threadIdx.x;

    numbers[idx] = ((seed + idx * 13) % 100) + 1;
}

void findWinner(int p1, int p2)
{
    printf("GPU 1 generated: %d\n", p1);
    printf("GPU 2 generated: %d\n", p2);

    if(p1 > p2)
        printf("Winner: GPU 1\n");

    else if(p2 > p1)
        printf("Winner: GPU 2\n");

    else
        printf("Result: Draw\n");
}

int main()
{
    int *d_numbers;
    int h_numbers[2];

    cudaMalloc((void**)&d_numbers, 2 * sizeof(int));

    srand(time(0));

    for(int round = 1; round <= 5; round++)
    {
        printf("\nRound %d\n", round);

        int seed = rand();

        generateNumbers<<<1,2>>>(d_numbers, seed);

        cudaMemcpy(h_numbers, d_numbers,
                   2*sizeof(int),
                   cudaMemcpyDeviceToHost);

        findWinner(h_numbers[0], h_numbers[1]);
    }

    cudaFree(d_numbers);

    return 0;
}
```
## Output
<img width="441" height="679" alt="WhatsApp Image 2026-05-22 at 3 29 05 PM" src="https://github.com/user-attachments/assets/8db6e868-ea94-4f0b-973b-69ca38f79c1c" />

## Result
The project successfully demonstrates a GPU vs GPU Number Battle Game using CUDA. It generates numbers using CUDA kernels, executes multiple rounds, and displays the winner using GPU parallel processing.
