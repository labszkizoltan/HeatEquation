
#include "HeatEquationKernels.cuh"


__global__ void HeatEquation_kernel(float3* target, float3* source, unsigned int gridSize, float deltaTime)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (gridSize * gridSize))
    {
        unsigned int x = i / gridSize;
        unsigned int y = i % gridSize;

        unsigned int x_next = (x == (gridSize - 1)) ? 0 : x + 1;
        unsigned int x_prev = (x == 0) ? (gridSize - 1) : x - 1;
        unsigned int y_next = (y == (gridSize - 1)) ? 0 : y + 1;
        unsigned int y_prev = (y == 0) ? (gridSize - 1) : y - 1;

        // write output vertex
        target[y * gridSize + x].y = source[y * gridSize + x].y + deltaTime * (source[y_next * gridSize + x].y + source[y_prev * gridSize + x].y + source[y * gridSize + x_next].y + source[y * gridSize + x_prev].y - 4.0f * source[y * gridSize + x].y);
    }
}



__global__ void HeatSource_kernel(float3* target, float3* source, unsigned int gridSize, float amplitude, float deltaTime)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (gridSize * gridSize))
    {
//        if (source[i].y > 0.0f) { target[i].y += (target[i].y < amplitude) ? min(source[i].y*deltaTime, amplitude-target[i].y) : 0.0f; }
//        if (source[i].y < 0.0f) { target[i].y += (target[i].y > 0.0) ? max(source[i].y*deltaTime, 0-target[i].y) : 0.0f; }

        target[i].y += (source[i].y > 0.0f) ? ((target[i].y < amplitude) ? min(source[i].y * deltaTime, amplitude - target[i].y) : 0.0f) : ((target[i].y > 0.0) ? max(source[i].y * deltaTime, 0 - target[i].y) : 0.0f);
    }
}



__global__ void SyncVertexBuffers_kernel(float3* target, float3* source, unsigned int gridSize)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (gridSize * gridSize))
    {
        target[i].y = source[i].y;
    }
}




