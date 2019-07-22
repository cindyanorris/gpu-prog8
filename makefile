NVCC = /usr/local/cuda-8.0/bin/nvcc
CC = g++
GENCODE_FLAGS = -arch=sm_30

#Optimization flags. Don't use this for debugging.
NVCCFLAGS = -c -m64 -O2 --compiler-options -Wall -Xptxas -O2,-v

#No optimizations. Debugging flags. Use this for debugging.
#NVCCFLAGS = -c -g -G -m64 --compiler-options -Wall

OBJS = blur.o wrappers.o h_blur.o d_blur.o
.SUFFIXES: .cu .o .h 
.cu.o:
	$(NVCC) $(CC_FLAGS) $(NVCCFLAGS) $(GENCODE_FLAGS) $< -o $@

all: blur generate

blur: $(OBJS)
	$(CC) $(OBJS) -L/usr/local/cuda/lib64 -lcuda -lcudart -ljpeg -o blur

blur.o: blur.cu wrappers.h h_blur.h d_blur.h

h_blur.o: h_blur.cu h_blur.h CHECK.h

d_blur.o: d_blur.cu d_blur.h CHECK.h

wrappers.o: wrappers.cu wrappers.h

generate: generate.c
	gcc -O2 generate.c -ljpeg -o generate

clean:
	rm generate blur *.o
