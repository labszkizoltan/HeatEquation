
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "src/window/window_context.h"
#include "src/controls/observer.h"
#include "src/rendering/coloured_mesh.h"
#include "src/rendering/shader.h"
#include "src/rendering/shader_sources/ShaderSources.h"
#include "src/utilities/Matrix_3D.h"


#include <stdio.h>


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

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // CUDA part over, lets try setting up a window

    MyWindow appWindow(windowWidth, windowHeight, "HeatEquation"); std::cout << glfwGetError(NULL) << "\n";
    glfwSetWindowPos(appWindow.GetWindow(), 100, 200); std::cout << glfwGetError(NULL) << "\n";
    appWindow.SetMouseScrollCallback(mouse_scroll_callback); std::cout << glfwGetError(NULL) << "\n";

    Observer observer;
    appWindow.SetUserPointer(&observer);


    // create a coloured mesh, just a triangle
    std::vector<Vec3D> vertexAndColourData = { {0,0,0},{0,0,0}, {1,0,0},{1,0,0}, {0,1,0},{0,1,0} };
    ColouredMesh triangleMesh(vertexAndColourData, { 0,1,2 });

    // create a shader
    Shader myShader(VertexShader_Coloured, FragmentShader_Coloured);
    {
        myShader.Bind();
        myShader.UploadUniformFloat3("body_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        myShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        myShader.UploadUniformFloat("body_scale", 1.0f);
        myShader.UploadUniformFloat3("observer_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        myShader.UploadUniformMat3("observer_orientation", glm::mat3(1.0f));
        myShader.UploadUniformFloat("zoom_level", 1.0f);
        myShader.UploadUniformFloat("aspect_ratio", (float)windowWidth/(float)windowHeight);
    }


    float time = (float)glfwGetTime();
    float timeSpeed = 1.0f; // PARAMETER initial time speed
    float timestep = 0.0f; // timestep can be initialized like this, because its constructor takes in only one float, implicit cast is possible
    float lastFrameTime = 0.0f;


    // Game loop
    while (!glfwWindowShouldClose(appWindow.GetWindow()))
    {
        lastFrameTime = (float)glfwGetTime();
        appWindow.HandleUserInputs(observer, timestep*timeSpeed);
        observer.SetObserverInShader(myShader);

        glClearColor(0.2f, 0.1f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set the speed of the simulation, note that the quality of the update will be worse, as the timestep will be bigger
        SetTimeSpeed(appWindow, timeSpeed);

        myShader.UploadUniformFloat3("body_translation", glm::vec3(0.0f, 0.0f, 0.0f));
        myShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        myShader.UploadUniformFloat("body_scale", 1.0f);
        triangleMesh.Draw();

        myShader.UploadUniformFloat3("body_translation", glm::vec3(-2.0f, 0.0f, 3.0f));
        myShader.UploadUniformMat3("body_orientation", glm::mat3(1.0f));
        myShader.UploadUniformFloat("body_scale", 2.0f);
        triangleMesh.Draw();

        // Swap the screen buffers
        glfwSwapBuffers(appWindow.GetWindow());
        std::cout << glfwGetError(NULL) << "\n";

        timestep = (float)glfwGetTime() - lastFrameTime;
        //		timestep = 0.017f;
    }

    // Terminates GLFW, clearing any resources allocated by GLFW.
    glfwTerminate();





    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
