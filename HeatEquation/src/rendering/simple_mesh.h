#pragma once

#include "buffer.h"
#include "shader.h"
#include "src/controls/observer.h"
#include "Vector_3D.h"

// This class is rather a coloured mesh, not a scene

class SimpleMesh
{
public:

	SimpleMesh() = default;
	SimpleMesh(OpenGLVertexBuffer& vb, OpenGLIndexBuffer& ib);
	SimpleMesh(const std::vector<Vec3D>& vertexAndColorData, const std::vector<uint32_t>& indexData);
	SimpleMesh(const std::string& filename);
	~SimpleMesh();

	SimpleMesh(const SimpleMesh& other) = delete; // copy constructor
	SimpleMesh& operator=(const SimpleMesh& other) = delete; // copy assignment
	SimpleMesh(SimpleMesh&& other) noexcept; // move constructor
	SimpleMesh& operator=(SimpleMesh&& other) noexcept; // move assignment

	void Draw();

//private:
public:
	uint32_t m_VertexArray;
	OpenGLVertexBuffer m_VertexBuffer;
	OpenGLIndexBuffer m_IndexBuffer;

};





