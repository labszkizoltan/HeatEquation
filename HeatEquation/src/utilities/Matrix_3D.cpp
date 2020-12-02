
#include "Matrix_3D.h"

#include <math.h>








// By default construct a zero matrix
Mat_3D::Mat_3D() : f1(Vec3D()), f2(Vec3D()), f3(Vec3D()) {}

Mat_3D::Mat_3D(Vec3D x, Vec3D y, Vec3D z) : f1(x), f2(y), f3(z) {}

Mat_3D& Mat_3D::Transpose()
{
	Vec3D tmp = { f1.y, f1.z, f2.z };

	f1.y = f2.x;
	f1.z = f3.x;
	f2.z = f3.y;
	f2.x = tmp.x;
	f3.x = tmp.y;
	f3.y = tmp.z;

	return *this;
}

float Mat_3D::Trace()
{
	return (f1.x + f2.y + f3.z);
}

Mat_3D Transpose(Mat_3D m)
{
	return Mat_3D({ m.f1.x, m.f2.x, m.f3.x }, { m.f1.y, m.f2.y, m.f3.y }, { m.f1.z, m.f2.z, m.f3.z });
}

// member function operator overrides
Mat_3D& Mat_3D::operator+=(const Mat_3D& m)
{
	f1 += m.f1;
	f2 += m.f2;
	f3 += m.f3;
	return *this;
}

Mat_3D& Mat_3D::operator-=(const Mat_3D& m)
{
	f1 -= m.f1;
	f2 -= m.f2;
	f3 -= m.f3;
	return *this;
}

Mat_3D& Mat_3D::operator*=(float c)
{
	f1 *= c;
	f2 *= c;
	f3 *= c;
	return *this;
}

Mat_3D& Mat_3D::operator/=(float c)
{
	f1 /= c;
	f2 /= c;
	f3 /= c;
	return *this;
}

glm::mat3 Mat_3D::Glm()
{
	return glm::mat3(f1.Glm(), f2.Glm(), f3.Glm());
}



// Non member function operator overloads

std::ostream& operator<<(std::ostream& stream, const Mat_3D& m)
{
	stream << "\n(" << m.f1.x << ", " << m.f2.x << ", " << m.f3.x << ")";
	stream << "\n(" << m.f1.y << ", " << m.f2.y << ", " << m.f3.y << ")";
	stream << "\n(" << m.f1.z << ", " << m.f2.z << ", " << m.f3.z << ")";
	return stream;
}

Mat_3D operator+(const Mat_3D& m1, const Mat_3D& m2)
{
	return Mat_3D(m1.f1 + m2.f1, m1.f2 + m2.f2, m1.f3 + m2.f3);
}

Mat_3D operator-(const Mat_3D& m1, const Mat_3D& m2)
{
	return Mat_3D(m1.f1 - m2.f1, m1.f2 - m2.f2, m1.f3 - m2.f3);
}

Mat_3D operator*(const Mat_3D& m1, const Mat_3D& m2)
{
	// just type this out
	return Mat_3D(
		m1.f1*m2.f1.x + m1.f2*m2.f1.y + m1.f3*m2.f1.z,
		m1.f1*m2.f2.x + m1.f2*m2.f2.y + m1.f3*m2.f2.z,
		m1.f1*m2.f3.x + m1.f2*m2.f3.y + m1.f3*m2.f3.z
	);
}

// multiplication with a constant
Mat_3D operator*(const Mat_3D& m1, float c)
{
	return Mat_3D(m1.f1 * c, m1.f2 * c, m1.f3 * c);
}

Mat_3D operator*(float c, const Mat_3D& m1)
{
	return Mat_3D(m1.f1 * c, m1.f2 * c, m1.f3 * c);
}

// multiplication with a vector, aka applying the matrix on a column vector
Vec3D operator*(const Mat_3D& m1, const Vec3D v)
{
	return (m1.f1 * v.x + m1.f2 * v.y + m1.f3 * v.z);
}

Vec3D operator*(const Vec3D v, const Mat_3D& m1)
{
	return Vec3D(v*m1.f1, v*m1.f2, v*m1.f3);
}


// some special initializers:
Mat_3D Identity(float x)
{
	return Mat_3D({ x,0,0 }, { 0,x,0 }, { 0,0,x });
}

Mat_3D Diagonal(Vec3D v)
{
	return Mat_3D(Vec3D(v.x, 0, 0), Vec3D(0, v.y, 0), Vec3D(0, 0,v.z));
}

Mat_3D CrossProduct(Vec3D v)
{
	return Mat_3D({ 0,v.z,-v.y }, { -v.z,0,v.x }, { v.y,-v.x,0 });
}

Mat_3D DiadicProduct(Vec3D v1, Vec3D v2)
{
	return Mat_3D(v1 * v2.x, v1* v2.y, v1* v2.z);
}

// angle is the angle of rotation in radians,
// theta and phi are defining the axis of the rotation, by two angles:
// theta is the angle from the z axis
// phi is the polar angle
Mat_3D Rotation(float angle, float theta, float phi)
{
	if (angle == 0.0f) { return Identity(1.0f); }
	Vec3D n = { sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) };
	Mat_3D id = Identity(cos(angle)), diad = (1 - cos(angle))*DiadicProduct(n, n), crossprod = sin(angle)*CrossProduct(n);

	return id + diad + crossprod;
}

// angle is the rotation angle in radians, but the axis is specified by a vector instead of its polar coordinates
// this vector doesnt need to be normalized, as the function internally normalizes it
Mat_3D Rotation(float angle, Vec3D axis)
{
	if (angle == 0.0f) { return Identity(1.0f); }
	if (axis.length() == 0.0f) {
		return Identity(1.0f);
	}
	Vec3D n = axis / axis.length();
	Mat_3D id = Identity(cos(angle)), diad = (1 - cos(angle))*DiadicProduct(n, n), crossprod = sin(angle)*CrossProduct(n);

	return id + diad + crossprod;
}


