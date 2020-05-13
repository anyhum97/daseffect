#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "reflection.cu"

////////////////////////////////////////////////////////////////////////

static int Width;
static int Height;

static bool IsLoaded = false;

static cudaEvent_t start;
static cudaEvent_t stop;

////////////////////////////////////////////////////////////////////////

static Reflection<float> Buffer;
static Reflection<int> Frame;

static Reflection<float> MaxValueBuffer;
static Reflection<float> MinValueBuffer;
static Reflection<float> SumBuffer;

static Reflection<float> MaxValue;
static Reflection<float> MinValue;

static Reflection<float> Sum;

////////////////////////////////////////////////////////////////////////

__inline__ __device__ int Color(const int R, const int G, const int B)
{
	return (-16777216) | (R << 16) | (G << 8) | B;
}

////////////////////////////////////////////////////////////////////////

__inline__ __host__ __device__ unsigned int GetBufferIndex(const unsigned int dim, int x, int y, const unsigned int Width, const unsigned int Height)
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

__inline__ __host__ __device__ unsigned int GetFrameIndex(const unsigned int x, const unsigned int y, const unsigned int Width, const unsigned int Height)
{
	// Frame[Width][Height];

	return x*Height + y;
}

////////////////////////////////////////////////////////////////////////

typedef int (*ColorInterpretator)(float value, float maxValue, float minValue, float WaterLevel);

namespace ColorInterpretators
{
	////////////////////////////////////////////////////////////////////////

	__device__ int DefaultColor(float value, float MaxValue, float MinValue, float WaterLevel)
	{
		if(value == 0.0f)
		{
			return Color(255, 255, 255);
		}

		if(value < 0.0f)
		{
			int intensity = (int)(255.0f * (value / MinValue));
		
			return Color(0, 0, intensity);
		}
		else
		{
			int intensity = (int)(255.0f-255.0f * (value / MaxValue));
		
			return Color(intensity, intensity, intensity);
		}
	}

	////////////////////////////////////////////////////////////////////////

	__device__ ColorInterpretator Interpretators[] = 
	{
		DefaultColor,
	};

	const unsigned int Count = 1;

	char* Titles[] = 
	{
		"DefaultColor",
	};
}

////////////////////////////////////////////////////////////////////////

__global__ void CudaSample(float* Buffer, 
						   const unsigned int Width, 
						   const unsigned int Height, 
                           const float phaseSpeed)
{
	/// <<<Width, Height>>>

    const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block >= Width || thread >= Height)
    {
        return;
    }

	const float laplacian = Buffer[GetBufferIndex(1, block+1, thread, Width, Height)] + 
		                    Buffer[GetBufferIndex(1, block-1, thread, Width, Height)] +
		                    Buffer[GetBufferIndex(1, block, thread+1, Width, Height)] + 
		                    Buffer[GetBufferIndex(1, block, thread-1, Width, Height)] - 4.0f * 
		                    Buffer[GetBufferIndex(1, block, thread, Width, Height)];

	Buffer[GetBufferIndex(2, block, thread, Width, Height)] = 2.0f*Buffer[GetBufferIndex(1, block, thread, Width, Height)] + phaseSpeed*laplacian;
}

__global__ void PushBuffers(float* Buffer)
{
	///	<<<1, 1>>>


}

////////////////////////////////////////////////////////////////////////

__global__ void ReCountPart1(float* Buffer, 
							 float* MaxValueBuffer, 
							 float* MinValueBuffer, 
							 float* SumBuffer, 
							 const unsigned int Width, 
							 const unsigned int Height)
{
	/// <<<Width, 1>>>

	const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block >= Width || thread > 0)
    {
        return;
    }

	////////////////////////////////////////////////////////////////////////

	float max = -FLT_MAX;
	float min = FLT_MAX;

	float sum = 0.0f;

	for(int i=0; i<Height; ++i)
	{
		const float value = Buffer[GetBufferIndex(1, block, i, Width, Height)];

		if(value < min)
		{
			min = value;
		}

		if(value > max)
		{
			max = value;
		}

		sum += value;
	}

	MinValueBuffer[block] = min;
	MaxValueBuffer[block] = max;

	SumBuffer[block] = sum;
}

__global__ void ReCountPart2(float* Buffer, 
							 float* MaxValueBuffer, 
							 float* MinValueBuffer, 
							 float* SumBuffer, 
							 float* MaxValue, 
							 float* MinValue, 
							 float* Sum, 
							 const unsigned int Width, 
							 const unsigned int Height)
{
	/// <<<1, 1>>>

	const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block || thread)
    {
        return;
    }

	////////////////////////////////////////////////////////////////////////

	float max = -FLT_MAX;
	float min = FLT_MAX;

	float sum = 0.0f;

	for(int i=0; i<Width; ++i)
	{
		if(MaxValueBuffer[i] > max)
		{
			max = MaxValueBuffer[i];
		}

		if(MinValueBuffer[i] < min)
		{
			min = MinValueBuffer[i];
		}

		sum += SumBuffer[i];
	}

	MaxValue[0] = max;
	MinValue[0] = min;

	Sum[0] = sum;
}

////////////////////////////////////////////////////////////////////////

__global__ void CudaFrame(float* Buffer, 
						  int* Frame,
						  float* MaxValue, 
						  float* MinValue, 						  
						  const float WaterLevel,
						  const unsigned int InterpretatorIndex,
						  const unsigned int Width, 
						  const unsigned int Height)
{
	/// <<<Width, Height>>>

    const unsigned int block = blockIdx.x;
    const unsigned int thread = threadIdx.x;

    if(block >= Width || thread >= Height)
    {
        return;
    }

	const float value = Buffer[GetBufferIndex(1, block, thread, Width, Height)];

	Frame[GetFrameIndex(block, thread, Width, Height)] = ColorInterpretators::Interpretators[InterpretatorIndex](value, MaxValue[0], MinValue[0], WaterLevel);
}

void ReCount()
{
	ReCountPart1<<<Width, 1>>>(Device(Buffer), 
							   Device(MaxValueBuffer), 
							   Device(MinValueBuffer), 
							   Device(SumBuffer), 
							   Width, 
							   Height);

	ReCountPart2<<<1, 1>>>(Device(Buffer), 
						   Device(MaxValueBuffer), 
						   Device(MinValueBuffer), 
						   Device(SumBuffer), 
						   Device(MaxValue), 
						   Device(MinValue), 
						   Device(Sum), 
						   Width, 
						   Height);
}

////////////////////////////////////////////////////////////////////////

static void CudaMalloc(int Width, int Height)
{
	cudaSetDevice(0);

	Buffer = Malloc<float>(3*Width*Height);

	Frame = Malloc<int>(Width*Height);

	MaxValueBuffer = Malloc<float>(Width);
	MinValueBuffer = Malloc<float>(Width);

	SumBuffer = Malloc<float>(Width);

	MaxValue = Malloc<float>(1);
	MinValue = Malloc<float>(1);

	Sum = Malloc<float>(1);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

void CudaFree()
{
	Free(Buffer);
	Free(Frame);

	Free(MaxValueBuffer);
	Free(MinValueBuffer);

	Free(SumBuffer);

	IsLoaded = false;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

bool CudaStart(int width, int height)
{
	if(IsLoaded)
	{
		CudaFree();
	}

	if(width < 3 || height < 3 || height > 1024)
	{
		return false;
	}

	Width = width;
	Height = height;
	
	CudaMalloc(width, height);

	IsLoaded = true;

	return IsValid(Buffer);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

bool CudaSetState(float* buffer, int width, int height)
{
	if(buffer == nullptr || width < 3 || height < 3)
	{
		return false;
	}

	if(width != Width || height != Height)
	{
		CudaStart(width, height);
	}

	if(!IsValid(Buffer) || !IsLoaded)
	{
		return false;
	}

	const unsigned int size = 2*width*height*sizeof(float);

	memcpy(Host(Buffer), buffer, size);

	return Send(Buffer);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

bool SetDefaultState()
{
	if(!IsValid(Buffer) || !IsLoaded)
	{
		return false;
	}

	Host(Buffer)[GetBufferIndex(0, Width >> 1, Height >> 1, Width, Height)] = 1.0f;
	Host(Buffer)[GetBufferIndex(1, Width >> 1, Height >> 1, Width, Height)] = 1.0f;

	return Send(Buffer);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

int GetCurrentFrame(int* frame, int ColorInterpretatorIndex, float WaterLevel)
{
	if(!IsLoaded || !IsValid(Buffer) || !IsValid(Frame))
	{
		return -1;
	}

	////////////////////////////////////////////////////////////////////////

	if(ColorInterpretatorIndex > ColorInterpretators::Count)
	{
		ColorInterpretatorIndex = 0;
	}

	ColorInterpretator Selected = ColorInterpretators::Interpretators[ColorInterpretatorIndex];

	////////////////////////////////////////////////////////////////////////
	
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	////////////////////////////////////////////////////////////////////////

	ReCount();
	
	if(Height <= 1024)
	{
		CudaFrame<<<Width, Height>>>(Device(Buffer), 
									 Device(Frame),
									 Device(MaxValue), 
									 Device(MinValue), 									 
									 WaterLevel, 
									 ColorInterpretatorIndex, 
									 Width, 
									 Height);
									 
	}

	if(!Receive(Frame))
	{
		return -1;
	}

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 0;

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	memcpy(frame, Host(Frame), Frame.size);

	return Frame.size;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

int CudaCalc(float phaseSpeed)
{
	if(!IsLoaded || !IsValid(Buffer) || !IsValid(Frame))
	{
		return -1;
	}

	////////////////////////////////////////////////////////////////////////

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	////////////////////////////////////////////////////////////////////////

	if(Height <= 1024)
	{
		CudaSample<<<Width, Height>>>(Device(Buffer), Width, Height, phaseSpeed);
	}

	////////////////////////////////////////////////////////////////////////

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float time = 0;

	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return (int)(time+0.5f);
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

int GetColorInterpretatorCount()
{
	return ColorInterpretators::Count;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

int GetColorInterpretatorTitle(char* str, int ColorInterpretatorIndex)
{
	if(ColorInterpretatorIndex > ColorInterpretators::Count)
	{
		return 0;
	}

	int len = strlen(ColorInterpretators::Titles[ColorInterpretatorIndex]);
	memcpy(str, ColorInterpretators::Titles[ColorInterpretatorIndex], len);	
	return len;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport)

bool GetCudaStatus(int width, int height)
{
	bool Status = IsLoaded;

	Status = Status && IsValid(Buffer);
	Status = Status && IsValid(Frame);

	Status = Status && IsValid(MaxValueBuffer);
	Status = Status && IsValid(MinValueBuffer);
	Status = Status && IsValid(SumBuffer);

	Status = Status && IsValid(MaxValue);
	Status = Status && IsValid(MinValue);
	Status = Status && IsValid(Sum);

	Status = Status && width == Width;
	Status = Status && height == Height;

	return Status;
}

////////////////////////////////////////////////////////////////////////

extern "C" __declspec(dllexport) 

float GetSum()
{
	if(Receive(Sum))
	{
		return Host(Sum)[0];
	}

	return 0.0f;
}

////////////////////////////////////////////////////////////////////////





