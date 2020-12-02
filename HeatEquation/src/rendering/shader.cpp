
#include "renderer.h"
#include "shader.h"

#include "gtc/type_ptr.hpp"


std::string ParseShader(const std::string& filepath)
{
	std::ifstream stream(filepath);
	std::stringstream ss;

	std::string line;
	while (getline(stream, line))
	{
		ss << line << '\n';
	}

	return ss.str();
}

static uint32_t CompileShader(uint32_t type, const std::string& source)
{
	uint32_t id = glCreateShader(type);
	// source should exist at this point
	const char* src = source.c_str(); // equivalent to &source[0]
	GLCall(glShaderSource(id, 1, &src, nullptr));
	GLCall(glCompileShader(id));

	// TODO: Error handling
	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		int length;
		glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char* message = (char*)alloca(length * sizeof(char));

		glGetShaderInfoLog(id, length, &length, message);
		std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader\n";
		std::cout << message << std::endl;
		glDeleteShader(id);
		return 0;
	}

	return id;
}

Shader::Shader()
	: m_RendererID(0)//, m_UniformLocations()
{
	std::cout << "Default Shader was created.\n";
}

// Use like this: Shader my_shader(ParseShader("folder/folder2/vertex_src.glsl"), ParseShader("folder/folder2/fragment_src.glsl"));
Shader::Shader(const std::string & vertexSrc, const std::string & fragmentSrc)
{
	m_RendererID = glCreateProgram();
	uint32_t vs = CompileShader(GL_VERTEX_SHADER, vertexSrc);
	uint32_t fs = CompileShader(GL_FRAGMENT_SHADER, fragmentSrc);

	GLCall(glAttachShader(m_RendererID, vs));
	GLCall(glAttachShader(m_RendererID, fs));
	GLCall(glLinkProgram(m_RendererID));
	GLCall(glValidateProgram(m_RendererID));

	GLCall(glDeleteShader(vs));
	GLCall(glDeleteShader(fs));
}

// // copy constructor
// Shader::Shader(const Shader & other)
// 	: m_RendererID(other.m_RendererID), m_UniformLocations(other.m_UniformLocations)
// {
// 	std::cout << "Shader::Shader(const Shader & other) is being used!" << std::endl;
// }
// 
// // copy assignment
// // This functions probably would be best to avoid. When the shader is copied, two instances of this class will hold the same renderer ID,
// // and when of the objects are destroyed, it will destroy the shader in the GPU. But the other object still holds a reference to that GPU program.
// Shader & Shader::operator=(const Shader & other)
// {
// 	std::cout << "Shader has been copy assigned! \n";
// 	glDeleteProgram(m_RendererID);
// 	return *this = Shader(other);
// }

// move constructor
Shader::Shader(Shader && other) noexcept
{
	std::cout << "Shader has been move constructed! \n";
	m_RendererID = other.m_RendererID;
//	m_UniformLocations = other.m_UniformLocations;

	other.m_RendererID = 0;
}

// move assignment
Shader & Shader::operator=(Shader && other) noexcept
{
	std::cout << "Shader has been move assigned! \n";
	if (this != &other)
	{
		glDeleteProgram(m_RendererID);
//		m_UniformLocations.clear();

		m_RendererID = other.m_RendererID;
//		m_UniformLocations = other.m_UniformLocations;

		other.m_RendererID = 0;
	}

	return *this;
}

Shader::~Shader()
{
	glDeleteProgram(m_RendererID);
}




void Shader::Bind() const
{
	glUseProgram(m_RendererID);
}

void Shader::Unbind() const
{
	glUseProgram(0);
}


void Shader::UploadUniformInt(const std::string & name, const int value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform1i(m_UniformLocations[name], value);
		glUniform1i(loc, value);
	}
	else
	{
		std::cout << "Shader::UploadUniformInt() : uniform called \"" << name << "\" does not exist in shader , or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformIntArray(const std::string & name, int * values, uint32_t count)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform1iv(m_UniformLocations[name], count, values);
		glUniform1iv(loc, count, values);
	}
	else
	{
		std::cout << "Shader::UploadUniformIntArray() : uniform called \"" << name << "\" does not exist in shader , or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}


void Shader::UploadUniformFloat(const std::string & name, const float value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform1f(m_UniformLocations[name], value);
		glUniform1f(loc, value);
	}
	else
	{
		std::cout << "Shader::UploadUniformFloat() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}


void Shader::UploadUniformFloat2(const std::string & name, const glm::vec2 & value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform2f(m_UniformLocations[name], value.x, value.y);
		glUniform2f(loc, value.x, value.y);
	}
	else
	{
		std::cout << "Shader::UploadUniformFloat2() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformFloat3(const std::string & name, const glm::vec3 & value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform3f(m_UniformLocations[name], value.x, value.y, value.z);
		glUniform3f(loc, value.x, value.y, value.z);
	}
	else
	{
		std::cout << "Shader::UploadUniformFloat3() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformFloat4(const std::string & name, const glm::vec4 & value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniform4f(m_UniformLocations[name], value.x, value.y, value.z, value.w);
		glUniform4f(loc, value.x, value.y, value.z, value.w);
	}
	else
	{
		std::cout << "Shader::UploadUniformFloat4() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformFloatArray(const std::string & name, float * values, uint32_t count)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
	//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
		//		glUniform1fv(m_UniformLocations[name], count, values);
		glUniform1fv(loc, count, values);
	}
	else
	{
		std::cout << "Shader::UploadUniformFloatArray() : uniform called \"" << name << "\" does not exist in shader , or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformMat3(const std::string & name, const glm::mat3 & value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if (loc != -1)
	{
//		glUniformMatrix3fv(m_UniformLocations[name], 1, GL_FALSE, glm::value_ptr(value));
		glUniformMatrix3fv(loc, 1, GL_FALSE, glm::value_ptr(value));
	}
	else
	{
		std::cout << "Shader::UploadUniformMat3() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

void Shader::UploadUniformMat4(const std::string & name, const glm::mat4 & value)
{
	GLint loc = glGetUniformLocation(m_RendererID, name.c_str());
//	if (m_UniformLocations.find(name) != m_UniformLocations.end())
	if(loc != -1)
	{
//		glUniformMatrix4fv(m_UniformLocations[name], 1, GL_FALSE, glm::value_ptr(value));
		glUniformMatrix4fv(loc, 1, GL_FALSE, glm::value_ptr(value));
	}
	else
	{
		std::cout << "Shader::UploadUniformMat4() : uniform called \"" << name << "\" does not exist in shader, or hasn't been added with Shader::SearchAndAddUniform()\n";
	}
}

 void Shader::SearchAndAddUniform(const std::string & name)
 {
// 	m_UniformLocations[name] = glGetUniformLocation(m_RendererID, name.c_str());
 }




