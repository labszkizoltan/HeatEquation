
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"

#include "src/window/window_context.h"
#include "src/controls/observer.h"
#include "src/rendering/coloured_mesh.h"
#include "src/rendering/simple_mesh.h"
#include "src/rendering/shader.h"
#include "src/rendering/shader_sources/ColouredShaderSources.h"
#include "src/rendering/shader_sources/SimpleShaderSources.h"
#include "src/rendering/grid_factory/CreateGridData.h"

#include "src/utilities/Matrix_3D.h"


#include <stdio.h>

#include <cuda_gl_interop.h> // this has to be included after some other headers, not sure which ones, havent tried all possibilities so I put this to the end, but at first this caused a compile error!!!

const int windowWidth = 1200;
const int windowHeight = 800;

void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    Observer* obsPtr = (Observer*)glfwGetWindowUserPointer(window);
    if (yoffset > 0) { obsPtr->ZoomIn(1.1f); } // PARAMETER zoom multiplier
    else if (yoffset < 0) { obsPtr->ZoomOut(1.1f); }
}

void SetTimeSpeed(MyWindow& appWindow, float& timeSpeed)
{
    if (appWindow.IsKeyPressed(GLFW_KEY_SPACE)) { timeSpeed = 0.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_1)) { timeSpeed = 1.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_2)) { timeSpeed = 2.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_3)) { timeSpeed = 8.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_4)) { timeSpeed = 16.0f; }
}


__global__ void simple_vbo_kernel(float3* pos_1, float3* pos_2, unsigned int gridSize, float time)
{
//    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    // prime numbers from https://primes.utm.edu/lists/small/10000.txt
    int quasiRand = (101119*(threadIdx.x+threadIdx.y+(int)time)+82031)%7993;
    float qRand = (float)quasiRand / (float)7993 - 0.5f;


    // write output vertex
    pos_1[y * gridSize + x].y += 0.5f*qRand;
    pos_1[y * gridSize + x].y *= 0.997f;
}


__global__ void HeatEquation_kernel(float3* target, float3* source, unsigned int gridSize, float deltaTime)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

//    if (i < (gridSize * gridSize-1) && i>1)
    if (i < (gridSize*gridSize))
    {
        unsigned int x = i / gridSize;
        unsigned int y = i % gridSize;

        unsigned int x_next = (x == (gridSize - 1)) ? 0 : x + 1;
        unsigned int x_prev = (x == 0) ? (gridSize - 1) : x - 1;
        unsigned int y_next = (y == (gridSize - 1)) ? 0 : y + 1;
        unsigned int y_prev = (y == 0) ? (gridSize - 1) : y - 1;

        // write output vertex
        target[y * gridSize + x].y = source[y * gridSize + x].y + deltaTime * (source[y_next * gridSize + x].y + source[y_prev * gridSize + x].y + source[y * gridSize + x_next].y + source[y * gridSize + x_prev].y - 4.0f * source[y * gridSize + x].y);
    }
}


__global__ void SyncVertexBuffers_kernel(float3* target, float3* source, unsigned int gridSize)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (gridSize * gridSize))
    {
        target[i].y = source[i].y;
    }
}





int main()
{
    uint32_t gridSize = 1000;
    uint32_t gridElements = gridSize * gridSize;
    float amplitude = 500.0f;

    // CUDA part over, lets try setting up a window

    MyWindow appWindow(windowWidth, windowHeight, "HeatEquation"); std::cout << glfwGetError(NULL) << "\n";
    glfwSetWindowPos(appWindow.GetWindow(), 100, 200); std::cout << glfwGetError(NULL) << "\n";
    appWindow.SetMouseScrollCallback(mouse_scroll_callback); std::cout << glfwGetError(NULL) << "\n";

    Observer observer;
    observer.translation = Vec3D(0.5f * (float)gridSize, 10.0f, -20.0f);// observer.TurnDown(0.5f);
    appWindow.SetUserPointer(&observer);

    // create a simple non coloured mesh, just a triangle
    std::vector<Vec3D> vertexData = GridFactory::CreateGridVertexData(gridSize, amplitude);
    std::vector<uint32_t> indexData = GridFactory::CreateGridIndexData(gridSize);
    SimpleMesh GridMesh_1(vertexData, indexData);
    SimpleMesh GridMesh_2(vertexData, indexData);

    Shader simpleShader(VertexShader_Simple, FragmentShader_Simple);
    {
        simpleShader.Bind();
        simpleShader.UploadUniformFloat3("body_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        simpleShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("body_scale", 1.0f);
        simpleShader.UploadUniformFloat3("observer_translation", glm::vec3(0.0f, 5.0f, -10.0f));
        simpleShader.UploadUniformMat3("observer_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("zoom_level", 1.0f);
        simpleShader.UploadUniformFloat("aspect_ratio", (float)windowWidth / (float)windowHeight);
        simpleShader.UploadUniformFloat("amplitude", amplitude);
    }


    // Cuda functions that need to be called
//  cudaGraphicsGLRegisterBuffer // once after the vertex buffer has been created
//  cudaGraphicsMapResources // every time in the rendering loop
//  cudaGraphicsResourceGetMappedPointer // every time in the rendering loop
//  cudaGraphicsUnmapResources // every time in the rendering loop
//  cudaGraphicsUnregisterResource // once after the vertex buffer has been destroyed


    struct cudaGraphicsResource* cuda_vbo_resource_1;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_1, (GLuint)GridMesh_1.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));
    struct cudaGraphicsResource* cuda_vbo_resource_2;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_2, (GLuint)GridMesh_2.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));


    float time = (float)glfwGetTime();
    float timeSpeed = 0.0f; // PARAMETER initial time speed
    float timestep = 0.0f; // timestep can be initialized like this, because its constructor takes in only one float, implicit cast is possible
    float lastFrameTime = 0.0f;

    // Game loop
    while (!glfwWindowShouldClose(appWindow.GetWindow()))
    {
        lastFrameTime = (float)glfwGetTime();
//        appWindow.HandleUserInputs(observer, timestep*timeSpeed);
        appWindow.HandleUserInputs(observer, timestep);

        // Set the speed of the simulation, note that the quality of the update will be worse, as the timestep will be bigger
        SetTimeSpeed(appWindow, timeSpeed);

        observer.SetObserverInShader(simpleShader);
        
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --------------------- //
        // do the CUDA part here //

        float3 *dptr_1, *dptr_2;
        size_t num_bytes;

        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_1, 0));
        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_2, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr_1, &num_bytes, cuda_vbo_resource_1));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr_2, &num_bytes, cuda_vbo_resource_1));

        // launch kernel here
        HeatEquation_kernel <<<gridElements/256+1, 256>>> (dptr_1, dptr_2, gridSize, timestep*timeSpeed);
//        checkCudaErrors(cudaDeviceSynchronize());
        HeatEquation_kernel <<<gridElements/256+1, 256>>> (dptr_2, dptr_1, gridSize, timestep*timeSpeed);

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_1, 0));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_2, 0));

        // --------------------- //


        GridMesh_1.Draw();

        // Swap the screen buffers
        glfwSwapBuffers(appWindow.GetWindow());
        std::cout << glfwGetError(NULL) << "\n";

        timestep = (float)glfwGetTime() - lastFrameTime;
        //		timestep = 0.017f;
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_1));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_2));

    // Terminates GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();

    return 0;
}


