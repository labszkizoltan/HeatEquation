
#include "Vector_3D.h"
#include "simple_mesh.h"

#include "glad/glad.h"

SimpleMesh::SimpleMesh(OpenGLVertexBuffer& vb, OpenGLIndexBuffer& ib)
{
	glGenVertexArrays(1, &m_VertexArray);
	glBindVertexArray(m_VertexArray);
	m_VertexBuffer = std::move(vb);
	m_VertexBuffer.Bind();
	m_VertexBuffer.SetLayout({
		{ShaderDataType::Float3, "aPos"}
		});
	m_IndexBuffer = std::move(ib);
}

SimpleMesh::SimpleMesh(const std::vector<Vec3D>& vertexAndColorData, const std::vector<uint32_t>& indexData)
	: m_VertexArray(0), m_VertexBuffer((float*)&vertexAndColorData[0], vertexAndColorData.size() * sizeof(Vec3D)), m_IndexBuffer((uint32_t*)&indexData[0], indexData.size())
{
	glGenVertexArrays(1, &m_VertexArray);
	glBindVertexArray(m_VertexArray);

	m_VertexBuffer.SetLayout({
		{ShaderDataType::Float3, "aPos"}
		});

}

SimpleMesh::SimpleMesh(const std::string & filename)
	: m_VertexArray(0), m_VertexBuffer(), m_IndexBuffer()
{
	std::ifstream myfile(filename.c_str());
	int v_count = 0, i_count = 0;

	myfile >> v_count;
	myfile >> i_count;

	std::vector<Vec3D> vertexAndColorData;
	std::vector<uint32_t> indexData;
	vertexAndColorData.resize(v_count); // each vertex consists of one Vec3D, the vertex position
	indexData.resize(i_count);

	for (int i = 0; i < v_count; i++)
	{
		myfile >> vertexAndColorData[i].x;
		myfile >> vertexAndColorData[i].y;
		myfile >> vertexAndColorData[i].z;
	}

	for (int i = 0; i < i_count; i++)
		myfile >> indexData[i];

	myfile.close();
	
	glGenVertexArrays(1, &m_VertexArray);
	glBindVertexArray(m_VertexArray);

	m_VertexBuffer = std::move(OpenGLVertexBuffer((float*)&vertexAndColorData[0], vertexAndColorData.size() * sizeof(Vec3D)));
	m_VertexBuffer.Bind();
	m_VertexBuffer.SetLayout({
		{ShaderDataType::Float3, "aPos"}
		});

	m_IndexBuffer = std::move(OpenGLIndexBuffer((uint32_t*)&indexData[0], indexData.size()));
}

SimpleMesh::~SimpleMesh()
{
	glDeleteVertexArrays(1, &m_VertexArray);
	m_VertexBuffer.~OpenGLVertexBuffer();
	m_IndexBuffer.~OpenGLIndexBuffer();
}

// move constructor
SimpleMesh::SimpleMesh(SimpleMesh && other) noexcept
{
	glDeleteVertexArrays(1, &m_VertexArray);
	m_VertexBuffer.~OpenGLVertexBuffer();
	m_IndexBuffer.~OpenGLIndexBuffer();

	m_VertexArray = other.m_VertexArray;
	m_VertexBuffer = std::move(other.m_VertexBuffer);
	m_IndexBuffer = std::move(other.m_IndexBuffer);

	other.m_VertexArray = 0;
}

// move assignment
SimpleMesh & SimpleMesh::operator=(SimpleMesh && other) noexcept
{
	if (this != &other)
	{
		glDeleteVertexArrays(1, &m_VertexArray);
		m_VertexBuffer.~OpenGLVertexBuffer();
		m_IndexBuffer.~OpenGLIndexBuffer();

		m_VertexArray = other.m_VertexArray;
		m_VertexBuffer = std::move(other.m_VertexBuffer);
		m_IndexBuffer = std::move(other.m_IndexBuffer);

		other.m_VertexArray = 0;
	}

	return *this;
}

void SimpleMesh::Draw()
{
	glBindVertexArray(m_VertexArray);
	m_VertexBuffer.SetLayout();
	m_IndexBuffer.Bind();
	glDrawElements(GL_TRIANGLES, m_IndexBuffer.m_Count, GL_UNSIGNED_INT, nullptr);
}
