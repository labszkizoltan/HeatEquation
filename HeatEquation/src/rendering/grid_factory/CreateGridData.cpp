
//#pragma once

#include "CreateGridData.h"

namespace GridFactory
{


	std::vector<Vec3D> CreateGridVertexData(int n, float amplitude)
	{
		std::vector<Vec3D> gridData;
		gridData.resize(n*n);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				gridData[n*i+j] = Vec3D(float(i), amplitude*(float)rand()/((float)RAND_MAX), float(j));
			}
		}

		return gridData;
	}

	std::vector<uint32_t> CreateGridIndexData(int n)
	{
		std::vector<uint32_t> indexData;
		indexData.resize((n-1)*(n-1)*6); // (n-1)*(n-1) is the number of quads in the grid, 6 is the number of indices per quad

		for (int i = 0; i < n-1; i++)
		{
			for (int j = 0; j < n-1; j++)
			{
				indexData[6*((n-1)*i+j) + 0] = 0   + n*i+j;
				indexData[6*((n-1)*i+j) + 1] = 1   + n*i+j;
				indexData[6*((n-1)*i+j) + 2] = n   + n*i+j;
				indexData[6*((n-1)*i+j) + 3] = 1   + n*i+j;
				indexData[6*((n-1)*i+j) + 4] = n   + n*i+j;
				indexData[6*((n-1)*i+j) + 5] = n+1 + n*i+j;
			}
		}

		return indexData;
	}










}
