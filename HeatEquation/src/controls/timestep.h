#pragma once


class Timestep
{
public:
	Timestep(float time = 0.0f);
	operator float() const; // this is like a static cast operator that lets us to cast this object implicitely

	float GetSeconds() const;
	float GetMiliSeconds() const;

private:
	float m_Time = 0.0f; // stores in seconds

};



