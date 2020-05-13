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
static Reflection<float> Frame;

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

	Frame = Malloc<float>(Width*Height);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

void CudaFree()
{
	Free(Buffer);
	Free(Frame);

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

bool CudaSetState(float* input, int width, int height)
{
	if(input == nullptr || width < 3 || height < 3)
	{
		return false;
	}

	if(width != Width || height != Height)
	{
		CudaStart(width, height);
	}

	if(!IsValid(Buffer))
	{
		return false;
	}

	const unsigned int size = 2*width*height*sizeof(float);

	memcpy(Host(Buffer), input, size);

	return Send(Buffer);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

bool GetCurrentFrame(float* frame)
{
	Frame.host[0] = 1.23f;
	Frame.host[1] = -2.25f;
	Frame.host[2] = 7.14f;

	Send(Frame);

	if(Receive(Frame))
	{
		memcpy(frame, Host(Frame), Frame.size);

		return true;
	}

	return false;
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











