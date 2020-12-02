#pragma once

#include "src/utilities/Vector_3D.h"
#include <vector>


namespace GridFactory
{
	std::vector<Vec3D> CreateGridVertexData(int n, float amplitude);
	std::vector<uint32_t> CreateGridIndexData(int n);



}









