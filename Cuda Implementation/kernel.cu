#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "reflection.cu"

////////////////////////////////////////////////////////////////////////

static int Width;
static int Height;

static bool IsLoaded = false;

////////////////////////////////////////////////////////////////////////

static Reflection<float> Buffer;

////////////////////////////////////////////////////////////////////////

__inline__ __device__ unsigned int GetBufferIndex(const unsigned int dim, int x, int y, const unsigned int Width, const unsigned int Height)
{
    if(x < 0)
	{
		x = x % Width + Width;
	}

	if(x >= Width)
	{
		x = x % Width;
	}

    if(y < 0)
	{
		y = y % Height + Height;
	}

	if(y >= Height)
	{
		y = y % Height;
	}

	// Buffer[3][Width][Height];

	return dim*Width*Height + x*Height + y;
}

////////////////////////////////////////////////////////////////////////

__global__ void CudaSample(float* value)
{
	/// <<<1, 1>>>

    const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block || thread)
    {
        return;
    }

	for(int i=0; i<16; ++i)
	{
		++value[0];
	}
}

////////////////////////////////////////////////////////////////////////

static void CudaMalloc(int Width, int Height)
{
	cudaSetDevice(0);

	Buffer = Malloc<float>(3*Width*Height);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

void CudaFree()
{
	Free(Buffer);

	IsLoaded = false;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

int CudaStart(int width, int height)
{
	if(IsLoaded)
	{
		CudaFree();
	}

	Width = width;
	Height = height;
	
	CudaMalloc(width, height);

	IsLoaded = true;

	return IsValid(Buffer);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

float CudaCalc()
{
	CudaSample<<<1, 1>>>(Device(Buffer));
	Receive(Buffer, 1);

	return Host(Buffer)[0];
}

////////////////////////////////////////////////////////////////////////











