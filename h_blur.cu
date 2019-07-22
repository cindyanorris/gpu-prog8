#include <stdio.h>
#include "h_blur.h"
#include "CHECK.h"
#include "wrappers.h"
#include "config.h"

void blurOnCPU(unsigned char * Pout, unsigned char * Pin, int height, 
               int width, int maskWidth, float * mask);
/*
   h_blur
   Performs the blur of an image on the CPU.

   Pout - array filled with the blur of each pixel.
   Pin - array that contains the color pixels.
   width and height - dimensions of the image.
   maskWidth - dimensions of the mask to be used
*/
float h_blur(unsigned char * Pout, unsigned char * Pin,
             int height, int width, int maskWidth)
{

    int i;
    cudaEvent_t start_cpu, stop_cpu;
    float cpuMsecTime = -1;
    int maskSize = maskWidth * maskWidth;

    //initialize the mask
    float * mask = (float *) Malloc(sizeof(float) * maskSize);
    for (i = 0; i < maskSize; i++) mask[i] = 1/(float)maskSize;

    //Use cuda functions to do the timing 
    //create event objects
    CHECK(cudaEventCreate(&start_cpu));
    CHECK(cudaEventCreate(&stop_cpu));
    //record the starting time
    CHECK(cudaEventRecord(start_cpu));

    blurOnCPU(Pout, Pin, height, width, maskWidth, mask);

    //record the ending time and wait for event to complete
    CHECK(cudaEventRecord(stop_cpu));
    CHECK(cudaEventSynchronize(stop_cpu));
    //calculate the elapsed time between the two events 
    CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
    return cpuMsecTime;
}

/*
   blurOnCPU
   Performs the blur of an image on the CPU.

   Pout - array filled with the blur of each pixel.
   Pin - array that contains the color pixels.
   width and height - dimensions of the image.
   maskWidth - dimensions of the mask to be used
   mask - mask to use in the convolution
*/
void blurOnCPU(unsigned char * Pout, unsigned char * Pin, int height, 
               int width, int maskWidth, float * mask)
{
    float redVal, greenVal, blueVal;
    int j, i, k, l, currRow, currCol;
    int halfMask = maskWidth >> 1;
    //calculate the row width of the input 
    int inRowWidth = CHANNELS * width;
    //calculate the row width of the output 
    int outRowWidth = CHANNELS * width;
    for (j = 0; j < height; ++j)
    {
        for (i = 0; i < width; i++)
        {
            //perform the 2D convolution
            redVal = greenVal = blueVal = 0;
            for (k = 0; k < maskWidth; k++)
            {
                currRow = j + (k - halfMask);
                if (currRow > -1 && currRow < height)
                {
                    for (l = 0; l < maskWidth; l++)
                    {
                        currCol = i + (l - halfMask);
                        if  (currCol > -1 && currCol < width)
                        {
                            redVal += Pin[currRow * inRowWidth + currCol * CHANNELS] * 
                                      mask[k * maskWidth + l];  
                            greenVal += Pin[currRow * inRowWidth + currCol * CHANNELS + 1] * 
                                        mask[k * maskWidth + l];  
                            blueVal += Pin[currRow * inRowWidth + currCol * CHANNELS + 2] * 
                                       mask[k * maskWidth + l];  
                        }
                    }
                }
            }
            //store the result
            Pout[j * outRowWidth + i * CHANNELS] = redVal;
            Pout[j * outRowWidth + i * CHANNELS + 1] = greenVal;
            Pout[j * outRowWidth + i * CHANNELS + 2] = blueVal;
        }
    }
}
