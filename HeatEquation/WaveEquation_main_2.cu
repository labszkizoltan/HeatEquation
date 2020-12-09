
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

#include "GlobalVariables.h"

//#include "src/cuda_kernels/HeatEquationKernels.cuh"
#include "src/cuda_kernels/WaveEquationKernels.cuh"

void mouse_scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    Observer* obsPtr = (Observer*)glfwGetWindowUserPointer(window);
    if (yoffset > 0) { obsPtr->ZoomIn(1.1f); } // PARAMETER zoom multiplier
    else if (yoffset < 0) { obsPtr->ZoomOut(1.1f); }
}

void SetTimeSpeed(MyWindow& appWindow, float& timeSpeed)
{
    if (appWindow.IsKeyPressed(GLFW_KEY_SPACE)) { timeSpeed = 0.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_1)) { timeSpeed = 0.1f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_2)) { timeSpeed = 1.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_3)) { timeSpeed = 2.0f; }
    if (appWindow.IsKeyPressed(GLFW_KEY_4)) { timeSpeed = 8.0f; }
}


int main()
{
    uint32_t gridSize = 100;
    uint32_t gridElements = gridSize * gridSize;
    float amplitude = 0.2f * (float)100;

    MyWindow appWindow(g_WindowWidth, g_WindowHeight, "WaveEquation"); std::cout << glfwGetError(NULL) << "\n";
    glfwSetWindowPos(appWindow.GetWindow(), 100, 200); std::cout << glfwGetError(NULL) << "\n";
    appWindow.SetMouseScrollCallback(mouse_scroll_callback); std::cout << glfwGetError(NULL) << "\n";

    Observer observer;
    observer.translation = Vec3D(0.5f * (float)gridSize, 10.0f, -20.0f);// observer.TurnDown(0.5f);
    appWindow.SetUserPointer(&observer);

    // Create the grids
    //std::vector<Vec3D> vertexData = GridFactory::CreateGridVertexData(gridSize, amplitude);

    std::vector<float> g_WaveEquationInitialCondition_100x100;
    g_WaveEquationInitialCondition_100x100.resize(100 * 100);
    for (int i = 0; i < 100; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            if (((i-25)*(i-25)+(j-25)*(j-25)) < 25)
                g_WaveEquationInitialCondition_100x100[100*i+j] = cos(0.63661977f/10.0f*(float)((i-25)*(i-25)+(j-25)*(j-25)));
        }
    }
//    g_WaveEquationInitialCondition_100x100[5050] = 1.0f;

    std::vector<float> flatGrid = GridFactory::MapAmplitudeFields(gridSize, g_FlatField_1x1);
    std::vector<float> initialConditionGrid = GridFactory::MapAmplitudeFields(gridSize, g_WaveEquationInitialCondition_100x100);

    std::vector<Vec3D> flatVertexData = GridFactory::CreateGridVertexData_with_amplitudes(gridSize, amplitude, flatGrid);
    std::vector<Vec3D> initialConditionVertexData = GridFactory::CreateGridVertexData_with_amplitudes(gridSize, amplitude, initialConditionGrid);
    std::vector<uint32_t> indexData = GridFactory::CreateGridIndexData(gridSize);
    SimpleMesh Grid_displacement(initialConditionVertexData, indexData);
    SimpleMesh Grid_velocity(flatVertexData, indexData);
    SimpleMesh Grid_acceleration(flatVertexData, indexData);

    Shader simpleShader(VertexShader_Simple, FragmentShader_Simple);
    {
        simpleShader.Bind();
        simpleShader.UploadUniformFloat3("body_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        simpleShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("body_scale", 1.0f);
        simpleShader.UploadUniformFloat3("observer_translation", glm::vec3(0.0f, 5.0f, -10.0f));
        simpleShader.UploadUniformMat3("observer_orientation", glm::mat3(1.0f));
        simpleShader.UploadUniformFloat("zoom_level", 1.0f);
        simpleShader.UploadUniformFloat("aspect_ratio", (float)g_WindowWidth / (float)g_WindowHeight);
        simpleShader.UploadUniformFloat("amplitude", amplitude/10.0f);
    }


    struct cudaGraphicsResource* cuda_vbo_resource_1;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_1, (GLuint)Grid_displacement.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));
    struct cudaGraphicsResource* cuda_vbo_resource_2;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_2, (GLuint)Grid_velocity.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));
    struct cudaGraphicsResource* cuda_vbo_resource_3;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource_3, (GLuint)Grid_acceleration.m_VertexBuffer.m_RendererID, cudaGraphicsMapFlagsNone));

    uint32_t blockSize = 320;

    float timeSpeed = 0.0f; // PARAMETER initial time speed
    float timestep = 0.005f; // timestep can be initialized like this, because its constructor takes in only one float, implicit cast is possible

    int counter = 0, draw_frequency = 10;

    // Game loop
    while (!glfwWindowShouldClose(appWindow.GetWindow()))
    {
//        appWindow.HandleUserInputs(observer, timestep);

        // Set the speed of the simulation, note that the quality of the update will be worse, as the timestep will be bigger
        SetTimeSpeed(appWindow, timeSpeed);

//        observer.SetObserverInShader(simpleShader);

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        if (timeSpeed > 0.0f)
        {

            // --------------------- //
            // do the CUDA part here //

            float3* dptr_r, * dptr_v, * dptr_a;
            size_t num_bytes;
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_1, 0));
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_2, 0));
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_resource_3, 0));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr_r, &num_bytes, cuda_vbo_resource_1));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr_v, &num_bytes, cuda_vbo_resource_2));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dptr_a, &num_bytes, cuda_vbo_resource_3));


            __global__ void UpdateDisplacement_kernel(float3 * displacement, float3 * velocity, unsigned int gridSize, float deltaTime);
            __global__ void UpdateVelocity_kernel(float3 * velocity, float3 * acceleration, unsigned int gridSize, float deltaTime);
            __global__ void UpdateAcceleration_kernel(float3 * acceleration, float3 * displacement, unsigned int gridSize, float deltaTime);

            // launch kernels here
            UpdateDisplacement_kernel <<<gridElements / blockSize + 1, blockSize >>> (dptr_r, dptr_v, gridSize, timestep*timeSpeed);
            UpdateVelocity_kernel     <<<gridElements / blockSize + 1, blockSize >>> (dptr_v, dptr_a, gridSize, timestep*timeSpeed);
            UpdateAcceleration_kernel <<<gridElements / blockSize + 1, blockSize >>> (dptr_a, dptr_r, gridSize, timestep*timeSpeed);

            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_1, 0));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_2, 0));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_resource_3, 0));

            // --------------------- //

        }

        if (counter > draw_frequency)
        {
            appWindow.HandleUserInputs(observer, timestep);
            observer.SetObserverInShader(simpleShader);
            Grid_displacement.Draw();
            glfwSwapBuffers(appWindow.GetWindow());
            counter = 0;
        }
        counter++;
        // Swap the screen buffers
//        glfwSwapBuffers(appWindow.GetWindow());
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_1));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_2));
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource_3));

    // Terminates GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();

    return 0;
}


