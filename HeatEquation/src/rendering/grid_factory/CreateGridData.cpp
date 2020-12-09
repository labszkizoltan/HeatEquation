
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
//				gridData[n * i + j] = Vec3D(float(i), amplitude * (float)rand() / ((float)RAND_MAX), float(j));
//				gridData[n * i + j] = Vec3D(float(i), amplitude * (((i < (n / 2)) && (j < (n / 2))) ? 1.0f : 0.0f), float(j));
				gridData[n * i + j] = Vec3D(float(i), amplitude * (((i<(3*n/4)) && (j<(3*n/4)) && (i>(n/4)) && (j>(n/4))) ? 1.0f : 0.0f), float(j));
			}
		}

		return gridData;
	}

	std::vector<Vec3D> CreateGridVertexData_with_amplitudes(int n, float amplitude, std::vector<float> amplitudes)
	{
		std::vector<Vec3D> gridData;
		gridData.resize(n*n);

		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				gridData[n*i+j] = Vec3D(float(i), amplitude*amplitudes[n*i+j], float(j));
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



	std::vector<float> MapAmplitudeFields(int target_size, std::vector<float> src)
	{
		std::vector<float> target; target.resize(target_size * target_size);
		int n_tar = (int)sqrt(target.size());
		int n_src = (int)sqrt(src.size());

		for (int i_tar = 0; i_tar < n_tar; i_tar++)
		{
			for (int j_tar = 0; j_tar < n_tar; j_tar++)
			{
				// get the countinuous coordinate in the target grid from the i/j indices
				float x_tar = (float)i_tar / (float)n_tar + 1.0f / (float)(2 * n_tar);
				float y_tar = (float)j_tar / (float)n_tar + 1.0f / (float)(2 * n_tar);
				// based on the continuous coordinates, get the corresponding indices from the source grid
				int i_src = floor(x_tar * (float)n_src);
				int j_src = floor(y_tar * (float)n_src);
				// put the values from the source grid to the target grid
				target[i_tar * n_tar + j_tar] = src[i_src * n_src + j_src];
			}
		}
		return target;
	}






}


















