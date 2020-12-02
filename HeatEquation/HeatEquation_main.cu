
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



cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



__global__ void simple_vbo_kernel(float3* pos, unsigned int gridSize, float time)
{
//    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;

    // prime numbers from https://primes.utm.edu/lists/small/10000.txt
    int quasiRand = (101119*(threadIdx.x+threadIdx.y+(int)time)+82031)%7993;
    float qRand = (float)quasiRand / (float)7993 - 0.5f;


    // write output vertex
//    pos[y * gridSize + x].y += 1.005f;
    pos[y * gridSize + x].y += 0.1f*qRand;
    pos[y * gridSize + x].y *= 0.997f;
}






int main()
{
    uint32_t gridSize = 32;

    // CUDA part over, lets try setting up a window

    MyWindow appWindow(windowWidth, windowHeight, "HeatEquation"); std::cout << glfwGetError(NULL) << "\n";
    glfwSetWindowPos(appWindow.GetWindow(), 100, 200); std::cout << glfwGetError(NULL) << "\n";
    appWindow.SetMouseScrollCallback(mouse_scroll_callback); std::cout << glfwGetError(NULL) << "\n";

    Observer observer;
    appWindow.SetUserPointer(&observer);

    // create a simple non coloured mesh, just a triangle
    std::vector<Vec3D> vertexData = GridFactory::CreateGridVertexData(gridSize, 10.0f);
    std::vector<uint32_t> indexData = GridFactory::CreateGridIndexData(gridSize);
    SimpleMesh SimpleTriangleMesh(vertexData, indexData);

    Shader simpleShader(VertexShader_Simple, FragmentShader_Simple);
    {
        simpleShader.Bind();
        simpleShader.UploadUniformFloat3("body_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        simpleShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("body_scale", 1.0f);
        simpleShader.UploadUniformFloat3("observer_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        simpleShader.UploadUniformMat3("observer_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("zoom_level", 1.0f);
        simpleShader.UploadUniformFloat("aspect_ratio", (float)windowWidth / (float)windowHeight);
        simpleShader.UploadUniformFloat("amplitude", 10.0f);
    }


    // Cuda functions that need to be called
//  cudaGraphicsGLRegisterBuffer // once after the vertex buffer has been created
//  cudaGraphicsMapResources // every time in the rendering loop
//  cudaGraphicsResourceGetMappedPointer // every time in the rendering loop
//  cudaGraphicsUnmapResources // every time in the rendering loop
//  cudaGraphicsUnregisterResource // once after the vertex buffer has been destroyed


    struct cudaGraphicsResource* cuda_vbo_resource;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, (GLuint)SimpleTriangleMesh.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));

    // map OpenGL buffer object for writing from CUDA
//    struct cudaGraphicsResource* cuda_vbo_resource;
////    cudaGraphicsResource** vbo_resource;
//    float3* dptr;
//    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
//    size_t num_bytes;
//    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes,
//        *vbo_resource));


    // execute the kernel

    // unmap buffer object
//    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));


    float time = (float)glfwGetTime();
    float timeSpeed = 1.0f; // PARAMETER initial time speed
    float timestep = 0.0f; // timestep can be initialized like this, because its constructor takes in only one float, implicit cast is possible
    float lastFrameTime = 0.0f;

    // Game loop
    while (!glfwWindowShouldClose(appWindow.GetWindow()))
    {
        lastFrameTime = (float)glfwGetTime();
        appWindow.HandleUserInputs(observer, timestep*timeSpeed);

        // Set the speed of the simulation, note that the quality of the update will be worse, as the timestep will be bigger
        SetTimeSpeed(appWindow, timeSpeed);

        observer.SetObserverInShader(simpleShader);
        
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // --------------------- //
        // do the CUDA part here //

        float3* dptr;
        size_t num_bytes;

        checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource, 0));
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource));

        // launch kernel here_
        simple_vbo_kernel <<<1, gridSize * gridSize>>> (dptr, gridSize, lastFrameTime);

        checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0));

        // --------------------- //


        SimpleTriangleMesh.Draw();

        // Swap the screen buffers
        glfwSwapBuffers(appWindow.GetWindow());
        std::cout << glfwGetError(NULL) << "\n";

        timestep = (float)glfwGetTime() - lastFrameTime;
        //		timestep = 0.017f;
    }


    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

    // Terminates GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();





    return 0;
}


