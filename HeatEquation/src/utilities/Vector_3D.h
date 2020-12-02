#pragma once
#include <iostream>

#include "glm.hpp"

// adding quaternions might be useful later on for representing rotations


struct Vec3D
{
	float x, y, z;
	Vec3D();
	Vec3D(float x, float y, float z);

	Vec3D& operator+=(const Vec3D& v);
	Vec3D& operator-=(const Vec3D& v);
	Vec3D& operator*=(float c);
	Vec3D& operator/=(float c);
	float length();
	float lengthSquare();
	glm::vec3 Glm();

	// Philip Nowell normalization (to map vectors onto a sphere): http://mathproofs.blogspot.com/2005/07/mapping-cube-to-sphere.html
	void PN_normalization();

};


std::ostream& operator<<(std::ostream& stream, const Vec3D& vec);

Vec3D operator+(const Vec3D& v1, const Vec3D& v2);

Vec3D operator-(const Vec3D& v1, const Vec3D& v2);

// Scalar product of vectors
float operator*(const Vec3D& v1, const Vec3D& v2);

// Multiply vector with a scalar
Vec3D operator*(const Vec3D& v1, float c);

// Multiply vector with a scalar, reversed order
Vec3D operator*(float c, const Vec3D& v1);

// Divide vector with a scalar
Vec3D operator/(const Vec3D& v1, float c);

// Vectorial cross product
Vec3D CrossProduct(const Vec3D& v1, const Vec3D& v2);














