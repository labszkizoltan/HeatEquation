#pragma once

#include <vector>
#include "Vector_3D.h"
#include "Matrix_3D.h"

struct Shape3D
{
	std::vector<Vec3D> vertices;
	std::vector<uint32_t> indices;

	Shape3D() = default;
	Shape3D(const std::string& filename);

	void merge(Shape3D shape_to_add);
	void simplify(float tolerance);

	void scale(float s);
	void translate(Vec3D r);
	void rotate(Mat_3D m);
	void center();
	void normalize();

	void write_to_file(const std::string& filename);
};


Shape3D CreateQuad();

Shape3D CreatePlane(uint32_t size_x, uint32_t size_y);

// size is just the resolution of the surfaces of the cube
Shape3D CreateCube(uint32_t size);

Shape3D CreateSphere(uint32_t size);


