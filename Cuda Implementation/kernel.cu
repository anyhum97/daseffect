#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

#include "reflection.cu"

int buffer[] = { 1, 2, 3, 4, 5, 6, 7 };

extern "C" __declspec(dllexport)

int test(int value)
{
	return value+1;
}

