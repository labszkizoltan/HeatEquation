
#include "shape_creator.h"

#include <fstream>

Shape3D::Shape3D(const std::string & filename)
{
	std::ifstream myfile(filename.c_str());
	int v_count=0, i_count=0;

	myfile >> v_count;
	myfile >> i_count;

	vertices.resize(v_count);
	indices.resize(i_count);

	for (int i = 0; i < v_count; i++)
	{
		myfile >> vertices[i].x;
		myfile >> vertices[i].y;
		myfile >> vertices[i].z;
	}

	for (int i = 0; i < i_count; i++)
		myfile >> indices[i];

	myfile.close();
}

void Shape3D::merge(Shape3D shape_to_add)
{
	int offset = vertices.size();

	for (int i = 0; i < shape_to_add.vertices.size(); i++)
	{
		vertices.push_back(shape_to_add.vertices[i]);
	}

	for (int i = 0; i < shape_to_add.indices.size(); i++)
	{
		indices.push_back(shape_to_add.indices[i] + offset);
	}
}


// if the same vertex position occurs multiple times and is being used in multiple triangles,
// this function will remove the duplicated vectors and will update the indices
void Shape3D::simplify(float tolerance)
{
	for (uint32_t i = 0; i < (vertices.size() - 1); i++)
	{
		Vec3D current_vec = vertices[i];
		uint32_t current_index = i;

		for (uint32_t j = i + 1; j < vertices.size(); j++)
		{
			Vec3D diff = current_vec - vertices[j];
			if (diff.length() < tolerance)
			{
				vertices.erase(vertices.begin() + j);
				// override the index buffer
				for (int k = 0; k < indices.size(); k++)
				{
					if (indices[k] == j) { indices[k] = current_index; }
					else if (indices[k] > j) { indices[k]--; } // when the j-th position is removed, the index array should be updated
				}
			}
		}
	}
}

void Shape3D::scale(float s)
{
	for (Vec3D& vert : vertices)
		vert *= s;
}

void Shape3D::translate(Vec3D r)
{
	for (Vec3D& vert : vertices)
		vert += r;
}

void Shape3D::rotate(Mat_3D m)
{
	for (Vec3D& vert : vertices)
		vert = m * vert;
}

void Shape3D::center()
{
	Vec3D avg = { 0.0f, 0.0f, 0.0f };

	for (Vec3D vert : vertices)
		avg += vert / (float)vertices.size();

	for (Vec3D& vert : vertices)
		vert -= avg;
}

void Shape3D::normalize()
{
	float max_length = 0.0f;

	for (Vec3D vert : vertices)
		max_length = vert.length() > max_length ? vert.length() : max_length;

	for (Vec3D& vert : vertices)
		vert /= max_length;
}

void Shape3D::write_to_file(const std::string& filename)
{
	std::ofstream myfile;
	myfile.open(filename.c_str());

	myfile << vertices.size() << " " << indices.size() << "\n";
	for(auto vec : vertices)
		myfile << vec.x << "\t" << vec.y << "\t" << vec.z << "\n";

	myfile << "\n";
	for (int i = 0; i < indices.size()/3; i++)
	{
		myfile << indices[3*i+0] << "\t" << indices[3 * i + 1] << "\t" << indices[3 * i + 2] << "\n";
	}

	myfile.close();
}


Shape3D CreateQuad()
{
	Shape3D quad;
	quad.vertices = {
		{-0.5f, -0.5f,  0.0f},
		{ 0.5f, -0.5f,  0.0f},
		{-0.5f,  0.5f,  0.0f},
		{ 0.5f,  0.5f,  0.0f}
	};

	quad.indices = {
		0, 1, 2,
		1, 3, 2
	};

	return quad;
}


Shape3D CreatePlane(uint32_t size_x, uint32_t size_y)
{
	Vec3D translate_x = { 1.0f, 0.0f, 0.0f };
	Vec3D translate_y = { 0.0f, 1.0f, 0.0f };

	Shape3D starting_quad = CreateQuad();

	for (int i = 0; i < size_x; i++)
	{
		for (int j = 0; j < size_y; j++)
		{
			if (!(i == 0 && j == 0))
			{
				Shape3D quad_to_add = CreateQuad();
				quad_to_add.simplify(0.0001f);
				quad_to_add.translate((float)j * translate_y + (float)i * translate_x);

				starting_quad.merge(quad_to_add);
				starting_quad.simplify(0.0001f);
			}
		}
	}

	return starting_quad;
}

// size is just the resolution of the surfaces of the cube
Shape3D CreateCube(uint32_t size)
{
	// face naming is consistent with rubik's cubes, L-left / R-right / U-upper / D-down / F-forward / B-backward
	Shape3D face_U = CreatePlane(size, size);
	face_U.center();
	face_U.normalize();

	Shape3D face_D = face_U;
	Shape3D face_F = face_U;
	Shape3D face_B = face_U;
	Shape3D face_L = face_U;
	Shape3D face_R = face_U;


	//	face_U.rotate();
	//	face_D.rotate();
//	face_L.rotate(Rotation(3.1415926535f / 2.0f, 3.1415926535f / 2.0f, 3.1415926535f / 2.0f));
//	face_R.rotate(Rotation(-3.1415926535f / 2.0f, 3.1415926535f / 2.0f, 3.1415926535f / 2.0f));
//	face_F.rotate(Rotation(3.1415926535f / 2.0f, 3.1415926535f / 2.0f, 0.0f));
//	face_B.rotate(Rotation(-3.1415926535f / 2.0f, 3.1415926535f / 2.0f, 0.0f));
//	face_D.rotate(Rotation(3.1415926535f / 2.0f, {}));

	face_L.rotate(Rotation(-3.1415926535f / 2.0f, {0,1,0}));
	face_R.rotate(Rotation(3.1415926535f / 2.0f, {0,1,0}));
	face_F.rotate(Rotation(3.1415926535f / 2.0f, {1,0,0}));
	face_B.rotate(Rotation(-3.1415926535f / 2.0f, {1,0,0}));
	face_D.rotate(Rotation(3.1415926535f, {1,0,0}));



	// 1/sqrt(2) = 0.7071067811865475
	float d = 0.707106781187f;
	face_F.translate({ 0.0f, -d, 0.0f });
	face_B.translate({ 0.0f,  d,  0.0f });
	face_L.translate({ -d, 0.0f, 0.0f });
	face_R.translate({ d, 0.0f, 0.0f });
	face_U.translate({ 0.0f, 0.0f,  d });
	face_D.translate({ 0.0f, 0.0f, -d });

	// Im leaving out the simplification step now, so it is easier to map texture coordinates to the vertices
	face_F.merge(face_B);
//	face_F.simplify(0.0001f);
	face_F.merge(face_L);
//	face_F.simplify(0.0001f);
	face_F.merge(face_R);
//	face_F.simplify(0.0001f);
	face_F.merge(face_U);
//	face_F.simplify(0.0001f);
	face_F.merge(face_D);
//	face_F.simplify(0.0001f);

	return face_F;
}

Shape3D CreateSphere(uint32_t size)
{
	Shape3D cube = CreateCube(size);
	cube.center();
	cube.normalize();

	// Scaling the cube so the distance between the center and faces is one
	cube.scale(sqrt(3.0f));

	for (int i = 0; i < cube.vertices.size(); i++)
	{
		cube.vertices[i].PN_normalization();
	}

	return cube;
}







