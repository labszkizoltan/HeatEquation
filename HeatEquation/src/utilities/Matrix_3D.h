#pragma once
#include <iostream>
#include "Vector_3D.h"


// The layout of this matrix struct is row major (it is determined by the matrix-vector and matrix-matrix multiplication rules)


struct Mat_3D
{
	Vec3D f1, f2, f3; // these are columns of the matrix
	Mat_3D();
	Mat_3D(Vec3D _x, Vec3D _y, Vec3D _z);

	Mat_3D& Transpose();
	float Trace();
	Mat_3D& operator+=(const Mat_3D& m);
	Mat_3D& operator-=(const Mat_3D& m);
	Mat_3D& operator*=(float c);
	Mat_3D& operator/=(float c);

	glm::mat3 Glm();

};




Mat_3D Transpose(Mat_3D m);

std::ostream& operator<<(std::ostream& stream, const Mat_3D& m);
Mat_3D operator+(const Mat_3D& m1, const Mat_3D& m2);
Mat_3D operator-(const Mat_3D& m1, const Mat_3D& m2);
Mat_3D operator*(const Mat_3D& m1, const Mat_3D& m2);
// multiplication with a constant
Mat_3D operator*(const Mat_3D& m1, float c);
Mat_3D operator*(float c, const Mat_3D& m1);
// multiplication with a vector, aka applying the matrix on a column vector
Vec3D operator*(const Mat_3D& m1, const Vec3D v);
Vec3D operator*(const Vec3D v, const Mat_3D& m1);
// some special initializers:
Mat_3D Identity(float x);
Mat_3D Diagonal(Vec3D v);
Mat_3D CrossProduct(Vec3D v);
Mat_3D DiadicProduct(Vec3D v1, Vec3D v2);

// angle is the angle of rotation in radians,
// theta and phi are defining the axis of the rotation, by two angles:
// theta is the angle from the z axis
// phi is the polar angle
Mat_3D Rotation(float angle, float theta, float phi);

// angle is the rotation angle in radians, but the axis is specified by a vector instead of its polar coordinates
// this vector doesnt need to be normalized, as the function internally normalizes it
Mat_3D Rotation(float angle, Vec3D axis);




















