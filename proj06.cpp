// 1. Program header

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

#include "cl.h"
#include "cl_platform.h"

// the matrix-width and the number of work-items per work-group:
// note: the matrices are actually MATWxMATW and the work group sizes are LOCALSIZExLOCALSIZE:

#ifndef MATW
#define MATW 1024
#endif

#ifndef LOCALSIZE
#define LOCALSIZE 8
#endif

// opencl objects:
cl_platform_id Platform;
cl_device_id Device;
cl_kernel Kernel;
cl_program Program;
cl_context Context;
cl_command_queue CmdQueue;

// do we want to print in csv file format?

#define CSV

float hA[MATW][MATW];
float hB[MATW][MATW];
float hC[MATW][MATW];

const char *CL_FILE_NAME = {"proj06.cl"};

// function prototypes:
void SelectOpenclDevice();
char *Vendor(cl_uint);
char *Type(cl_device_type);
void Wait(cl_command_queue);

int main(int argc, char *argv[])
{
#ifndef _OPENMP
	fprintf(stderr, "OpenMP is not enabled!\n");
	return 1;
#endif

	// see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE *fp;
#ifdef WIN32
	errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
	if (err != 0)
#else
	fp = fopen(CL_FILE_NAME, "r");
	if (fp == NULL)
#endif
	{
		fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
		return 1;
	}

	cl_int status; // returned status from opencl calls -- test against CL_SUCCESS

	// get the platform id and the device id:

	SelectOpenclDevice(); // sets the global variables Platform and Device

	// 2. allocate the host memory buffers:
	// allready done -- we did it as global variables instead of on the heap so could allocate them as a 2D array

	// initialize the input matrices:

	for (int i = 0; i < MATW; i++)
	{
		for (int j = 0; j < MATW; j++)
		{
			hA[i][j] = 1.0;
			hB[i][j] = 2.0;
		}
	}

	// 3. create an opencl context:

	Context = clCreateContext(NULL, 1, &Device, NULL, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateContext failed\n");

	// 4. create an opencl command queue:

	CmdQueue = clCreateCommandQueue(Context, Device, 0, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateCommandQueue failed\n");

	// 5. allocate the device memory buffers:

	size_t aSize = MATW * MATW * sizeof(float);
	size_t bSize = MATW * MATW * sizeof(float);
	int mw = MATW;
	size_t mwSize = sizeof(mw);
	size_t cSize = MATW * MATW * sizeof(float);

	cl_mem dA = clCreateBuffer(Context, CL_MEM_READ_ONLY, aSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (1)\n");

	cl_mem dB = clCreateBuffer(Context, CL_MEM_READ_ONLY, bSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (2)\n");

	cl_mem dMW = clCreateBuffer(Context, CL_MEM_READ_ONLY, mwSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (3)\n");

	cl_mem dC = clCreateBuffer(Context, CL_MEM_WRITE_ONLY, cSize, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateBuffer failed (4)\n");

	// 6. enqueue the 3 commands to write the data from the host buffers to the device buffers:

	status = clEnqueueWriteBuffer(CmdQueue, dA, CL_FALSE, 0, aSize, hA, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");

	status = clEnqueueWriteBuffer(CmdQueue, dB, CL_FALSE, 0, bSize, hB, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");

	status = clEnqueueWriteBuffer(CmdQueue, dMW, CL_FALSE, 0, mwSize, &mw, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueWriteBuffer failed (3)\n");

	Wait(CmdQueue);

	// 7. read the kernel code from a file ...

	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	char *clProgramText = new char[fileSize + 1]; // leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if (n != fileSize)
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n);

	// ... and create the kernel program:

	char *strings[1];
	strings[0] = clProgramText;
	Program = clCreateProgramWithSource(Context, 1, (const char **)strings, NULL, &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateProgramWithSource failed\n");
	delete[] clProgramText;

	// 8. compile and link the kernel code:

	char *options = {(char *)""};
	status = clBuildProgram(Program, 1, &Device, options, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		size_t size;
		clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char *log = new cl_char[size];
		clGetProgramBuildInfo(Program, Device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete[] log;
	}

	// 9. create the kernel object:

	Kernel = clCreateKernel(Program, "MatrixMult", &status);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clCreateKernel failed\n");

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg(Kernel, 0, sizeof(cl_mem), &dA);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (1)\n");

	status = clSetKernelArg(Kernel, 1, sizeof(cl_mem), &dB);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (2)\n");

	status = clSetKernelArg(Kernel, 2, sizeof(cl_mem), &dMW);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (3)\n");

	status = clSetKernelArg(Kernel, 3, sizeof(cl_mem), &dC);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clSetKernelArg failed (4)\n");

	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = {MATW, MATW, 1};
	size_t localWorkSize[3] = {LOCALSIZE, LOCALSIZE, 1};

#ifndef CSV
	fprintf(stderr, "Number of Work Groups = %5d x %5d\n", MATW / LOCALSIZE, MATW / LOCALSIZE);
#endif

	Wait(CmdQueue);

	double time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(CmdQueue, Kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);

	Wait(CmdQueue);
	double time1 = omp_get_wtime();

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer(CmdQueue, dC, CL_FALSE, 0, cSize, hC, 0, NULL, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clEnqueueReadBuffer failed\n");

	Wait(CmdQueue);

#ifdef CSV
	fprintf(stderr, "%8d , %6d , %10.2lf\n",
					MATW * MATW, LOCALSIZE * LOCALSIZE, (double)MATW * (double)MATW * (double)MATW / (time1 - time0) / 1000000000.);
#else
	fprintf(stderr, "Matrix Size: %6d x %6d , Work Elements: %4d x %4d , GigaMultsPerSecond: %10.2lf\n",
					MATW, MATW, LOCALSIZE, LOCALSIZE, (double)MATW * (double)MATW * (double)MATW / (time1 - time0) / 1000000000.);
#endif

	// 13. clean everything up:

	clReleaseKernel(Kernel);
	clReleaseProgram(Program);
	clReleaseCommandQueue(CmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dMW);
	clReleaseMemObject(dC);

	return 0;
}

// wait until all queued tasks have taken place:

void Wait(cl_command_queue queue)
{
	cl_event wait;
	cl_int status;

	status = clEnqueueMarker(queue, &wait);
	if (status != CL_SUCCESS)
		fprintf(stderr, "Wait: clEnqueueMarker failed\n");

	status = clWaitForEvents(1, &wait);
	if (status != CL_SUCCESS)
		fprintf(stderr, "Wait: clWaitForEvents failed\n");
}

// vendor ids:
#define ID_AMD 0x1002
#define ID_INTEL 0x8086
#define ID_NVIDIA 0x10de

void SelectOpenclDevice()
{
	// select which opencl device to use:
	// priority order:
	//	1. a gpu
	//	2. an nvidia or amd gpu
	//	3. an intel gpu
	//	4. an intel cpu

	int bestPlatform = -1;
	int bestDevice = -1;
	cl_device_type bestDeviceType;
	cl_uint bestDeviceVendor;
	cl_int status; // returned status from opencl calls
								 // test against CL_SUCCESS

	// find out how many platforms are attached here and get their ids:

	cl_uint numPlatforms;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (1)\n");

	cl_platform_id *platforms = new cl_platform_id[numPlatforms];
	status = clGetPlatformIDs(numPlatforms, platforms, NULL);
	if (status != CL_SUCCESS)
		fprintf(stderr, "clGetPlatformIDs failed (2)\n");

	for (int p = 0; p < (int)numPlatforms; p++)
	{
		// find out how many devices are attached to each platform and get their ids:

		cl_uint numDevices;

		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		cl_device_id *devices = new cl_device_id[numDevices];
		status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
		if (status != CL_SUCCESS)
			fprintf(stderr, "clGetDeviceIDs failed (2)\n");

		for (int d = 0; d < (int)numDevices; d++)
		{
			cl_device_type type;
			cl_uint vendor;
			size_t sizes[3] = {0, 0, 0};

			clGetDeviceInfo(devices[d], CL_DEVICE_TYPE, sizeof(type), &type, NULL);

			clGetDeviceInfo(devices[d], CL_DEVICE_VENDOR_ID, sizeof(vendor), &vendor, NULL);

			// select:

			if (bestPlatform < 0) // not yet holding anything -- we'll accept anything
			{
				bestPlatform = p;
				bestDevice = d;
				Platform = platforms[bestPlatform];
				Device = devices[bestDevice];
				bestDeviceType = type;
				bestDeviceVendor = vendor;
			}
			else // holding something already -- can we do better?
			{
				if (bestDeviceType == CL_DEVICE_TYPE_CPU) // holding a cpu already -- switch to a gpu if possible
				{
					if (type == CL_DEVICE_TYPE_GPU) // found a gpu
					{																// switch to the gpu we just found
						bestPlatform = p;
						bestDevice = d;
						Platform = platforms[bestPlatform];
						Device = devices[bestDevice];
						bestDeviceType = type;
						bestDeviceVendor = vendor;
					}
				}
				else // holding a gpu -- is a better gpu available?
				{
					if (bestDeviceVendor == ID_INTEL) // currently holding an intel gpu
					{																	// we are assuming we just found a bigger, badder nvidia or amd gpu
						bestPlatform = p;
						bestDevice = d;
						Platform = platforms[bestPlatform];
						Device = devices[bestDevice];
						bestDeviceType = type;
						bestDeviceVendor = vendor;
					}
				}
			}
		}
		delete[] devices;
	}
	delete[] platforms;

	if (bestPlatform < 0)
	{
		fprintf(stderr, "I found no OpenCL devices!\n");
		exit(1);
	}
	else
	{
#ifndef CSV
		fprintf(stderr, "I have selected Platform #%d, Device #%d: ", bestPlatform, bestDevice);
		fprintf(stderr, "Vendor = %s, Type = %s\n", Vendor(bestDeviceVendor), Type(bestDeviceType));
#endif
	}
}

char *
Vendor(cl_uint v)
{
	switch (v)
	{
	case ID_AMD:
		return (char *)"AMD";
	case ID_INTEL:
		return (char *)"Intel";
	case ID_NVIDIA:
		return (char *)"NVIDIA";
	}
	return (char *)"Unknown";
}

char *
Type(cl_device_type t)
{
	switch (t)
	{
	case CL_DEVICE_TYPE_CPU:
		return (char *)"CL_DEVICE_TYPE_CPU";
	case CL_DEVICE_TYPE_GPU:
		return (char *)"CL_DEVICE_TYPE_GPU";
	case CL_DEVICE_TYPE_ACCELERATOR:
		return (char *)"CL_DEVICE_TYPE_ACCELERATOR";
	}
	return (char *)"Unknown";
}
