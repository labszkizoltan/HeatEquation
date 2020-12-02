#include "Vector_2D.h"
#include <math.h>

Vec2D::Vec2D() : x(0), y(0) {}

Vec2D::Vec2D(float x, float y) : x(x), y(x) {}

Vec2D & Vec2D::operator+=(const Vec2D & v)
{
	this->x += v.x;
	this->y += v.y;
	return *this;
}

Vec2D & Vec2D::operator-=(const Vec2D & v)
{
	this->x -= v.x;
	this->y -= v.y;
	return *this;
}

Vec2D & Vec2D::operator*=(float c)
{
	this->x *= c;
	this->y *= c;
	return *this;
}

Vec2D & Vec2D::operator/=(float c)
{
	this->x /= c;
	this->y /= c;
	return *this;
}

float Vec2D::length()
{
	return sqrt(x*x+y*y);
}

glm::vec2 Vec2D::Glm()
{
	return glm::vec2{x,y};
}


/************************
*	Operator overloads	*
*************************/

std::ostream& operator<<(std::ostream& stream, const Vec2D& vec)
{
	stream << "(" << vec.x << ", " << vec.y << ")";
	return stream;
}

Vec2D operator+(const Vec2D& v1, const Vec2D& v2)
{
	return Vec2D(v1.x + v2.x, v1.y + v2.y);
}

Vec2D operator-(const Vec2D& v1, const Vec2D& v2)
{
	return Vec2D(v1.x - v2.x, v1.y - v2.y);
}

// Scalar product of vectors
float operator*(const Vec2D& v1, const Vec2D& v2)
{
	return (v1.x*v2.x + v1.y*v2.y);
}

// Multiply vector with a scalar
Vec2D operator*(const Vec2D& v1, float c)
{
	return Vec2D(v1.x*c, v1.y*c);
}

// Multiply vector with a scalar, reversed order
Vec2D operator*(float c, const Vec2D& v1)
{
	return Vec2D(v1.x*c, v1.y*c);
}

// Divide vector with a scalar
Vec2D operator/(const Vec2D& v1, float c)
{
	return Vec2D(v1.x / c, v1.y / c);
}








