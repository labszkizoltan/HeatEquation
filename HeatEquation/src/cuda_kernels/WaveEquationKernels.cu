
#include "WaveEquationKernels.cuh"

__global__ void WaveEquation_kernel(float3* slice1, float3* slice2, float3* slice3, unsigned int gridSize, float deltaTime)
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
        slice3[y*gridSize+x].y = 0.01f * deltaTime*deltaTime * (slice1[y_next*gridSize+x].y + // 1.0f is the square of the wave speed
            slice1[y_prev*gridSize+x].y +
            slice1[y*gridSize+x_next].y +
            slice1[y*gridSize+x_prev].y -
            4.0f * slice1[y*gridSize+x].y) +
            2.0f * slice2[y*gridSize+x].y - slice1[y*gridSize+x].y;
    }
}



// __global__ void WaveSource_kernel(float3* target, float3* source, unsigned int gridSize, float amplitude, float deltaTime)
// {
//     unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < (gridSize * gridSize))
//     {
//         //        if (source[i].y > 0.0f) { target[i].y += (target[i].y < amplitude) ? min(source[i].y*deltaTime, amplitude-target[i].y) : 0.0f; }
//         //        if (source[i].y < 0.0f) { target[i].y += (target[i].y > 0.0) ? max(source[i].y*deltaTime, 0-target[i].y) : 0.0f; }
// 
//         target[i].y += (source[i].y > 0.0f) ? ((target[i].y < amplitude) ? min(source[i].y * deltaTime, amplitude - target[i].y) : 0.0f) : ((target[i].y > 0.0) ? max(source[i].y * deltaTime, 0 - target[i].y) : 0.0f);
//     }
// }






__global__ void UpdateDisplacement_kernel(float3* displacement, float3* velocity, unsigned int gridSize, float deltaTime)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (gridSize * gridSize))
    {
        displacement[i].y += velocity[i].y * deltaTime;
    }
}

__global__ void UpdateVelocity_kernel(float3* velocity, float3* acceleration, unsigned int gridSize, float deltaTime)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (gridSize * gridSize))
    {
        velocity[i].y += acceleration[i].y * deltaTime;
        velocity[i].y *= 0.9999f; // put in some friction like force
    }
}

__global__ void UpdateAcceleration_kernel(float3* acceleration, float3* displacement, unsigned int gridSize, float deltaTime)
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

        // acceleration is just the laplacian of the displacement field (times c**2, but make it to be 1)
        acceleration[y * gridSize + x].y = (displacement[y_next * gridSize + x].y + // 1.0f is the square of the wave speed
            displacement[y_prev * gridSize + x].y +
            displacement[y * gridSize + x_next].y +
            displacement[y * gridSize + x_prev].y -
            4.0f * displacement[y * gridSize + x].y);
    }
}








