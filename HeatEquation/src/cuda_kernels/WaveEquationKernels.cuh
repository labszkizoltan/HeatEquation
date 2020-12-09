
#pragma once

__global__ void WaveEquation_kernel(float3* slice1, float3* slice2, float3* slice3, unsigned int gridSize, float deltaTime);
// __global__ void WaveSource_kernel(float3* target, float3* source, unsigned int gridSize, float amplitude, float deltaTime);

__global__ void UpdateDisplacement_kernel(float3* displacement, float3* velocity, unsigned int gridSize, float deltaTime);
__global__ void UpdateVelocity_kernel(float3* velocity, float3* acceleration, unsigned int gridSize, float deltaTime);
__global__ void UpdateAcceleration_kernel(float3* acceleration, float3* displacement, unsigned int gridSize, float deltaTime);





