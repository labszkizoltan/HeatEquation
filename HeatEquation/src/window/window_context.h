#pragma once

//#include <iostream>

// GLAD
//#include <glad/glad.h>

// GLFW
#include <glfw3.h>

#include "src/controls/observer.h"

class MyWindow
{
public:
	MyWindow(int width, int height, std::string name = "DefaultName");

	void SetKeyCallback(GLFWkeyfun callback);
	void SetMouseButtonCallback(GLFWmousebuttonfun callback);
	void SetCursorPositionCallback(GLFWcursorposfun callback);
	void SetMouseScrollCallback(GLFWcursorposfun callback);

	bool IsKeyPressed(int keycode);
	bool IsMouseButtonPressed(int button);
	std::pair<float, float> GetMousePosition();
	std::pair<float, float> GetMousePositionFromCenter();
	
//	float GetMouseX();
//	float GetMouseY();

	void HandleUserInputs(Observer& obs, float timestep);

	void SetUserPointer(void* userPtr);
	void* GetUserPointer(void* userPtr);

	GLFWwindow* GetWindow() { return m_Window; }

private:
	GLFWwindow* m_Window;
	float m_Width, m_Height;
};



