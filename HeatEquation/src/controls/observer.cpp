
#include "observer.h"


// -------------------- //
// ----- Observer ----- //
// -------------------- //

// the zoom level should be around 2.5 to fill the entire screen with the skybox
Observer::Observer() : translation(Vec3D()), orientation(Identity(1.0f)), zoom_level(2.5f) {}

Observer::Observer(Vec3D v, Mat_3D m, float zoom) : translation(v), orientation(m), zoom_level(zoom) {}

void Observer::SetObserverInShader(Shader& shader)
{
	shader.Bind();
	shader.UploadUniformFloat3("observer_translation", translation.Glm());
	shader.UploadUniformMat3("observer_orientation", orientation.Glm());
	shader.UploadUniformFloat("zoom_level", zoom_level);
//	shader.UploadUniformFloatArray("obs_param", (float*)this, 13); // hardcoded number: 3 float - translation, 9 float - orientation, 1 float - zoom level; 13 float in total
}


void Observer::MoveForward(float distance)
{
	translation += distance * orientation.f3;
}

void Observer::MoveBackward(float distance)
{
	translation -= distance * orientation.f3;
}

void Observer::MoveLeft(float distance)
{
	translation -= distance * orientation.f1;
}

void Observer::MoveRight(float distance)
{
	translation += distance * orientation.f1;
}

void Observer::MoveUp(float distance)
{
	translation += distance * orientation.f2;
}

void Observer::MoveDown(float distance)
{
	translation -= distance * orientation.f2;
}

// ------------------- //
// ----- Turning ----- //
// ------------------- //

void Observer::Turn(Vec3D axis, float angle)
{
	orientation = Rotation(angle, axis) * orientation;
}

void Observer::TurnRight(float angle)
{
	orientation = Rotation(angle, orientation.f2) * orientation;
}

void Observer::TurnLeft(float angle)
{
	orientation = Rotation(-angle, orientation.f2) * orientation;
}

void Observer::TurnUp(float angle)
{
	orientation = Rotation(angle, orientation.f1) * orientation;
}

void Observer::TurnDown(float angle)
{
	orientation = Rotation(-angle, orientation.f1) * orientation;
}

void Observer::TurnClockwise(float angle)
{
	orientation = Rotation(-angle, orientation.f3) * orientation;
}

void Observer::TurnAntiClockwise(float angle)
{
	orientation = Rotation(angle, orientation.f3) * orientation;
}

void Observer::ZoomIn(float multiplier)
{
	zoom_level *= zoom_level > 1000.0f ? 1.0f : multiplier; // PARAMETER maxZoomLevel
}

void Observer::ZoomOut(float multiplier)
{
	zoom_level /= zoom_level < 0.5f ? 1.0f : multiplier;
}






