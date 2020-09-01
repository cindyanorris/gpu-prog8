#include <sys/stat.h>
#include <stdlib.h>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>
#include "wrappers.h"
#include "h_blur.h"
#include "d_blur.h"
#include "config.h"  //defines BLOCKDIM, CHANNELS, MASK_MAX_HEIGHT, MASK_MAX_WIDTH

//prototypes for functions in this file 
void parseCommandArgs(int, char **, char **, int *, int *);
void printUsage();
void readJPGImage(char *, unsigned char **, int *, int *);
void writeJPGImage(char *, unsigned char *, int, int);
char * buildFilename(char *, const char *, int);
void compare(unsigned char *, unsigned char *, int, int);
void checkColor(const char *, int, int, int, int, unsigned char *, unsigned char *);

/*
    main 
    Opens the jpg file and reads the contents.  Uses the CPU
    and the GPU to compute the blur.  Compares the CPU and GPU
    results.  If save desired, writes the results to output files.  
    Outputs the time of each.
*/
int main(int argc, char * argv[])
{
    unsigned char * Pin, * h_Pout, * d_Pout;
    char * fileName, * outFileNm;
    int numBytes, width, height, maskWidth, saveOutput;
    float cpuTime, gpuTime;

    parseCommandArgs(argc, argv, &fileName, &maskWidth, &saveOutput);
    readJPGImage(fileName, &Pin, &width, &height);

    //calculate size of output array
    numBytes = width * height * CHANNELS * sizeof(unsigned char);

    //allocate space for both the host output and the device output
    h_Pout = (unsigned char *) Malloc(numBytes);
    d_Pout = (unsigned char *) Malloc(numBytes);

    printf("\nBlurring: %s, width: %d, height: %d\n", fileName, width, height);

    //use the CPU to perform the blur
    cpuTime = h_blur(h_Pout, Pin, height, width, maskWidth);
    free(Pin);
    printf("\tCPU time: %f msec\n", cpuTime);

    //use the GPU to perform the color 
    readJPGImage(fileName, &Pin, &width, &height);
    gpuTime = d_blur(d_Pout, Pin, height, width, maskWidth);
    free(Pin);

    //compare the CPU and GPU results
    compare(d_Pout, h_Pout, height, width);

    printf("\tGPU time: %f msec\n", gpuTime);
    printf("\tSpeedup: %f\n", cpuTime/gpuTime);

    //save images to output files if desired
    if (saveOutput)
    {
        outFileNm = buildFilename(fileName, "h_blur", maskWidth);
        writeJPGImage(outFileNm, h_Pout, width, height);
        free(outFileNm);
        outFileNm = buildFilename(fileName, "d_blur", maskWidth);
        writeJPGImage(outFileNm, d_Pout, width, height);
        free(outFileNm);
    }
    free(d_Pout);
    free(h_Pout);
    return EXIT_SUCCESS;
}

/* 
    compare
    This function takes two arrays of color pixel values.  One array
    contains pixel values calculated  by the GPU.  The other array contains
    color pixel values calculated by the CPU.  This function examines
    each pixel in the arrays to see that they match.

    d_Pout - pixel values calculated by GPU
    h_Pout - pixel values calculated by CPU
    height - height of image
    width - width of image
    
    Outputs an error message and exits program if the arrays differ.
*/
void compare(unsigned char * d_Pout, unsigned char * h_Pout, int height, int width)
{
    int i, j;
    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            checkColor("red", i, j, width, 0, d_Pout, h_Pout); 
            checkColor("green", i, j, width, 1, d_Pout, h_Pout); 
            checkColor("blue", i, j, width, 2, d_Pout, h_Pout); 
        }
    }
}

/* 
    checkColor
    This function takes two arrays of color pixel values.  One array
    contains pixel values calculated  by the GPU.  The other array contains
    color pixel values calculated by the CPU.  The function uses
    a specific row, column, and offset (to choose the channel) to
    see that the pixel colors are the same within a slight margin
    of error.

    color - string to use in error message if colors differ
    row - row containing pixel
    col - column containing the pixel
    width - width of image (row * CHANNELS is the row width)
    offset - offset into pixel to grab the color byte
    d_Pout - pixel values calculated by GPU
    h_Pout - pixel values calculated by CPU
    
    Outputs an error message and exits program if the arrays differ.
*/
void checkColor(const char * color, int row, int col, int width, 
                int offset, unsigned char * d_Pout, unsigned char * h_Pout)
{
    unsigned char gpuByte = d_Pout[row * width * CHANNELS + col + offset];
    unsigned char cpuByte = h_Pout[row * width * CHANNELS + col + offset];
    int diff = gpuByte - cpuByte;

    //GPU and CPU have different floating point standards so
    //the results could be slightly different
    if (abs(diff) > 1)
    {
        printf("%s blurred results don't match.\n", color);
        printf("CPU byte [%d, %d]: %d\n", row, col, cpuByte);
        printf("GPU byte [%d, %d]: %d\n", row, col, gpuByte);
        exit(EXIT_FAILURE);
    }
}


/* 
    writeJPGImage
    Writes a color jpg image to an output file.

    outfile - name of jpg file (ends with a .jpg extension)
    Pout - array of pixels
    width - width (x-dimension) of image
    height - height (y-dimension) of image
*/
void writeJPGImage(char * filename, unsigned char * Pout, 
                   int width, int height)
{
   struct jpeg_compress_struct cinfo;
   struct jpeg_error_mgr jerr;
   JSAMPROW rowPointer[1];

   //set up error handling
   cinfo.err = jpeg_std_error(&jerr);
   //initialize the compression object
   jpeg_create_compress(&cinfo);

   //open the output file
   FILE * fp;
   if ((fp = fopen(filename, "wb")) == NULL)
   {
     fprintf(stderr, "Can't open %s\n", filename);
     exit(1);
   }
   //initalize state for output to outfile
   jpeg_stdio_dest(&cinfo, fp);

   cinfo.image_width = width;    /* image width and height, in pixels */
   cinfo.image_height = height;
   cinfo.input_components = CHANNELS;   /* # of color components per pixel */
   cinfo.in_color_space = JCS_RGB;
   jpeg_set_defaults(&cinfo);
   jpeg_set_quality(&cinfo, 75, TRUE);

   //TRUE means it will write a complete interchange-JPEG file
   jpeg_start_compress(&cinfo, TRUE);

   while (cinfo.next_scanline < cinfo.image_height)
   {
      rowPointer[0] = &Pout[cinfo.next_scanline * width * CHANNELS];
      (void) jpeg_write_scanlines(&cinfo, rowPointer, 1);
   }
   jpeg_finish_compress(&cinfo);
   fclose(fp);
   jpeg_destroy_compress(&cinfo);
}

/*
    buildFilename
    This function builds a string by concatenating prefix,
    "M", maskWidth, and infile in that order.
    It is used by the program to build the output file names.
*/    
char * buildFilename(char * infile, const char * prefix, int maskWidth)
{
   int len = strlen(infile) + strlen(prefix) + 8;
   char * outfile = (char *) Malloc(sizeof(char *) * len);
   sprintf(outfile,"%sM%d%s", prefix, maskWidth, infile);
   return outfile;
}
   
/*
    readJPGImage
    This function opens a jpg file and reads the contents.  
    Each pixel consists of bytes for red, green, and blue.  r
    The array Pin is initialized to the pixel bytes.  width, height,
    are pointers to ints that are set to those values.
    filename - name of the .jpg file
*/
void readJPGImage(char * filename, unsigned char ** Pin, 
                  int * width, int * height) 
{
   unsigned long dataSize;             // length of the file
   int channels;                       //  3 =>RGB   4 =>RGBA 
   unsigned char * rowptr[1];          // pointer to an array
   unsigned char * jdata;              // data for the image
   struct jpeg_decompress_struct info; //for our jpeg info
   struct jpeg_error_mgr err;          //the error handler

   FILE * fp = fopen(filename, "rb"); //read binary
   if (fp == NULL)
   {
      fprintf(stderr, "Error reading file %s\n", filename);
      printUsage();
   }

   info.err = jpeg_std_error(& err);
   jpeg_create_decompress(&info);

   jpeg_stdio_src(&info, fp);
   jpeg_read_header(&info, TRUE);   // read jpeg file header
   jpeg_start_decompress(&info);    // decompress the file
 //set width and height
   (*width) = info.output_width;
   (*height) = info.output_height;
   channels = info.num_components;
   if (channels != CHANNELS)
   {
      fprintf(stderr, "%s is not an RGB jpeg image\n", filename);
      printUsage();
   }

   dataSize = (*width) * (*height) * channels;
   jdata = (unsigned char *)Malloc(dataSize);
   while (info.output_scanline < info.output_height) // loop
   {
      // Enable jpeg_read_scanlines() to fill our jdata array
      rowptr[0] = (unsigned char *)jdata +  // secret to method
                  channels * info.output_width * info.output_scanline;

      jpeg_read_scanlines(&info, rowptr, 1);
   }
   jpeg_finish_decompress(&info);   //finish decompressing
   jpeg_destroy_decompress(&info);
   fclose(fp);                      //close the file
   (*Pin) = jdata;
   return;
}

/*
    parseCommandArgs
    This function parses the command line arguments. The program can be executed 
    like this:
    ./blur [-m <maskWidth>] [-save]  <file>.jpg
    or
    ./blur <file>.jpg
    Any of the optional arguments can be omitted and they can be in any order.
    Mask width is set to the provided command line argument or to 3 if no 
    mask width is provided.  The mask width must be odd and less than the max mask width.  
    If the -save option is provided, output files
    are created to hold the output created by the CPU and the GPU.
    In addition, it checks to see if the last command line argument
    is a jpg file and sets (*fileNm) to argv[i] where argv[i] is the 
    name of the jpg file.  
*/
void parseCommandArgs(int argc, char * argv[], char ** fileNm, 
                      int * maskWidth, int * saveOutput)
{
    int fileIdx = argc - 1, mskW = 3, save = 0;
    struct stat buffer;

    for (int i = 1; i < argc - 1; i++)
    {
        //mask width
        if (strncmp("-m", argv[i], 3) == 0) 
        {
            mskW = atoi(argv[i+1]);
            if (mskW <= 0 || mskW > MAX_MASK_WIDTH) printUsage();
            if ((mskW % 2) != 1) printUsage();
            i++;
        //save output files
        } else if (strncmp("-save", argv[i], 6) == 0) 
        {
            save = 1;
        } else
            printUsage();
    } 

    //check the input file name (must end with .jpg)
    int len = strlen(argv[fileIdx]);
    if (len < 5) printUsage();
    if (strncmp(".jpg", &argv[fileIdx][len - 4], 4) != 0) printUsage();

    //stat function returns 1 if file does not exist
    if (stat(argv[fileIdx], &buffer)) printUsage();
    (*maskWidth) = mskW;
    (*fileNm) = argv[fileIdx];
    (*saveOutput) = save;
}

/*
    printUsage
    This function is called if there is an error in the command line
    arguments or if the .jpg file that is provided by the command line
    argument is improperly formatted.  It prints usage information and
    exits.
*/
void printUsage()
{
    printf("This application takes as input the name of a .jpg\n");
    printf("file containing a color image and creates a file\n");
    printf("containing a blurred color version of the file.\n");
    printf("\nusage: blur [-m <maskWidth>] [-save] <name>.jpg\n");
    printf("           <maskWidth> is the width of the mask created for blur.\n");
    printf("           <maskWidth> cannot be greater than %d.\n", MAX_MASK_WIDTH);
    printf("           <maskWidth> must be odd.\n");
    printf("           If the -m argument is omitted, the mask width\n");
    printf("           defaults to 3.\n");
    printf("           If the -save argument is provided, the outputs are saved.\n");
    printf("           host output file name: h_blurM<maskwidth><name>.jpg\n");
    printf("           device output file name: d_blurM<maskwidth>S<stride><name>.jpg\n");
    printf("           <name>.jpg is the name of the input jpg file.\n");
    printf("Examples:\n");
    printf("./blur color1200by800.jpg\n");
    printf("./blur -m 5 color1200by800.jpg\n");
    printf("./blur -m 5 -save color1200by800.jpg\n");
    exit(EXIT_FAILURE);
}
