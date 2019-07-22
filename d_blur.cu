#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "d_blur.h"
#include "CHECK.h"
#include "config.h"  //contains BLOCK_DIM, CHANNELS, MAX_MASK_WIDTH, MAX_MASK_HEIGHT
#include "wrappers.h"

//prototype for the kernel
__global__ void d_blurKernel(unsigned char *, unsigned char *,
                             int, int, int, int, float *);
/*
   d_blur
   Performs the blur of an image on the GPU.

   Pout - array that is filled with the blur of each pixel.
   Pin - array contains the color pixels to be blurred.
   width and height -  dimensions of the image.
   maskWidth - dimensions of the mask to be used
 
   Returns the amount of time it takes to perform the blur 
*/
float d_blur(unsigned char * Pout, unsigned char * Pin,
             int height, int width, int maskWidth)
{
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;
    unsigned char * d_Pin;
    size_t pitch;

    //generate the mask
    int maskSize = maskWidth * maskWidth;
    float * mask = (float *) Malloc(sizeof(float) * maskSize);
    for (int i = 0; i < maskSize; i++) mask[i] = 1/(float)maskSize;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    //This code handles allocating space for the input 
    //in the GPU memory.  The cudaMallocPitch function can
    //be used to allocate space for 2D data. Besides 
    //modifying d_Pin, it also modifies the pitch variable.
    //It will set pitch to K * width * CHANNELS where K >= 1.
    //Pitch will be a multiple of the burst size.
    CHECK(cudaMallocPitch((void **)&d_Pin, &pitch,
                          (size_t) (width * CHANNELS),
                          (size_t) height));
    //copy the rows from Pin to d_Pin.  Each row may
    //end with padding, due to the pitching.
    for (int i = 0; i < height; i++)
       CHECK(cudaMemcpy(&d_Pin[i * pitch], &Pin[i * width * CHANNELS],
             width * CHANNELS, cudaMemcpyHostToDevice));

    //Add the rest of the code here.  You still need to
    //allocate space for d_Pout and for d_mask and
    //copy the mask into d_d_mask.
  
    //You still need to define the grid and the block

//    d_blurKernel<<<grid, block>>>(d_Pout, d_Pin, height, width, 
//                                  pitch, maskWidth, d_mask);    

    //Copy the result into Pout.

    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*
   d_blurKernel
   Kernel code executed by each thread on its own data when the kernel is
   launched. Shared memory is used for both the mask and the pixels.
   Threads cooperate in loading the shared memory.  After the
   shared memory is filled, the convolution is performed.

   Pout - array that is filled with the blur of each pixel.
   Pin - array contains the color pixels to be blurred.
   width and height -  dimensions of the image in pixels.
   pitch - size of each row in bytes.
   maskWidth - dimensions of the mask to be used.
   mask - contains mask used for the convolution.
 
*/

__global__
void d_blurKernel(unsigned char * Pout, unsigned char * Pin, int height,
                  int width, int pitch, int maskWidth, float * mask)
{
    //add code here
    //need a 2D array for the shared pixels
    //need a 2D array for the shared mask
}
