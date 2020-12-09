
#pragma once

__global__ void HeatEquation_kernel(float3* target, float3* source, unsigned int gridSize, float deltaTime);
__global__ void HeatSource_kernel(float3* target, float3* source, unsigned int gridSize, float amplitude, float deltaTime);
__global__ void SyncVertexBuffers_kernel(float3* target, float3* source, unsigned int gridSize);






