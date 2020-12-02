#pragma once

#include "buffer.h"
#include "Vector_3D.h"

#include "shader.h"


class TexturedMesh
{
public:
//	TexturedMesh();
//	TexturedMesh(BodyType type);
	TexturedMesh(const std::string& vertexFilepath, uint32_t textureID); // use existing texture instead of creating one from a file
	TexturedMesh(const std::string& vertexFilepath, const std::string& textureFilepath);
	~TexturedMesh();

	void Draw();

private:
//	void Init(BodyType type);

private:
	uint32_t m_VertexArray;
	OpenGLVertexBuffer m_VertexBuffer; // layout should be: 3 floats - vertex position / 2 floats - texture coordinates
	OpenGLIndexBuffer m_IndexBuffer;
//	Shader m_Shader; // a shader per mesh is definitely not needed, maybe a pointer to an existing shader, so the proper shader can be bound before drawing
	uint32_t m_Texture;
};

