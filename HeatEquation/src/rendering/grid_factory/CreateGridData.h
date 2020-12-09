#pragma once

#include "src/utilities/Vector_3D.h"
#include <vector>


namespace GridFactory
{
	std::vector<Vec3D> CreateGridVertexData(int n, float amplitude);
	std::vector<Vec3D> CreateGridVertexData_with_amplitudes(int n, float amplitude, std::vector<float> amplitudes);
	std::vector<uint32_t> CreateGridIndexData(int n);

	std::vector<float> MapAmplitudeFields(int target_size, std::vector<float> src);

}



