#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>
//#include <unordered_map>

#include "glm.hpp"

typedef int GLint;

std::string ParseShader(const std::string& filepath);

class Shader
{
public:
//	Shader(const std::string& filepath);
	Shader();
	Shader(const std::string& vertexSrc, const std::string& fragmentSrc);

	Shader(const Shader& other) = delete;// copy constructor
	Shader& operator=(const Shader& other) = delete;// copy assignment
	Shader(Shader&& other) noexcept; // move constructor
	Shader& operator=(Shader&& other) noexcept; // move assignment
	~Shader();

	void Bind() const;
	void Unbind() const;

	void UploadUniformInt(const std::string& name, const int value);
	void UploadUniformIntArray(const std::string& name, int* values, uint32_t count);
	void UploadUniformFloat(const std::string& name, const float value);
	void UploadUniformFloat2(const std::string& name, const glm::vec2& value);
	void UploadUniformFloat3(const std::string& name, const glm::vec3& value);
	void UploadUniformFloat4(const std::string& name, const glm::vec4& value);
	void UploadUniformFloatArray(const std::string& name, float* values, uint32_t count);
	void UploadUniformMat3(const std::string& name, const glm::mat3& value);
	void UploadUniformMat4(const std::string& name, const glm::mat4& value);
	
	void SearchAndAddUniform(const std::string& name);

private:
	uint32_t m_RendererID;
//	std::map<std::string, GLint> m_UniformLocations;
//	std::unordered_map<std::string, GLint> m_UniformLocations;

	friend class Skybox;
};







