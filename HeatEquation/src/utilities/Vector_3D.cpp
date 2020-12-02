
#include "Vector_3D.h"

#include <math.h>


Vec3D::Vec3D() : x(0.0f), y(0.0f), z(0.0f) {}

Vec3D::Vec3D(float x, float y, float z) : x(x), y(y), z(z) {}

Vec3D& Vec3D::operator+=(const Vec3D& v)
{
	this->x += v.x;
	this->y += v.y;
	this->z += v.z;
	return *this;
}

Vec3D& Vec3D::operator-=(const Vec3D& v)
{
	this->x -= v.x;
	this->y -= v.y;
	this->z -= v.z;
	return *this;
}

Vec3D& Vec3D::operator*=(float c)
{
	this->x *= c;
	this->y *= c;
	this->z *= c;
	return *this;
}

Vec3D& Vec3D::operator/=(float c)
{
	this->x /= c;
	this->y /= c;
	this->z /= c;
	return *this;
}

float Vec3D::length()
{
	return sqrt(x*x + y * y + z * z);
}

float Vec3D::lengthSquare()
{
	return (x*x + y * y + z * z);
}

glm::vec3 Vec3D::Glm()
{
	return glm::vec3{x, y, z};
}

void Vec3D::PN_normalization()
{
	float x_new = x * sqrt(1.0f - y * y / 2.0f - z * z / 2.0f + y * y*z*z / 3.0f);
	float y_new = y * sqrt(1.0f - x * x / 2.0f - z * z / 2.0f + x * x*z*z / 3.0f);
	float z_new = z * sqrt(1.0f - y * y / 2.0f - x * x / 2.0f + y * y*x*x / 3.0f);

	x = x_new;
	y = y_new;
	z = z_new;
}


/************************
*	Operator overloads	*
*************************/

std::ostream& operator<<(std::ostream& stream, const Vec3D& vec)
{
	stream << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
	return stream;
}

Vec3D operator+(const Vec3D& v1, const Vec3D& v2)
{
	return Vec3D(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

Vec3D operator-(const Vec3D& v1, const Vec3D& v2)
{
	return Vec3D(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

// Scalar product of vectors
float operator*(const Vec3D& v1, const Vec3D& v2)
{
	return (v1.x*v2.x + v1.y*v2.y + v1.z*v2.z);
}

// Multiply vector with a scalar
Vec3D operator*(const Vec3D& v1, float c)
{
	return Vec3D(v1.x*c, v1.y*c, v1.z*c);
}

// Multiply vector with a scalar, reversed order
Vec3D operator*(float c, const Vec3D& v1)
{
	return Vec3D(v1.x*c, v1.y*c, v1.z*c);
}

// Divide vector with a scalar
Vec3D operator/(const Vec3D& v1, float c)
{
	return Vec3D(v1.x / c, v1.y / c, v1.z / c);
}

// Vectorial cross product
Vec3D CrossProduct(const Vec3D& v1, const Vec3D& v2)
{
	return Vec3D(v1.y*v2.z - v1.z*v2.y, v1.z*v2.x - v1.x*v2.z, v1.x*v2.y - v1.y*v2.x);
}




