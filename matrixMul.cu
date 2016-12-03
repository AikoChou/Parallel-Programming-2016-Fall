#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 2
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width);
__global__ void sMatrixMulKernel(float* Md, float* Nd, float* Pd, int Width);

int main(void){

	int width = 5;
	//Allocate and initialize the matrices M, N, P
	//I/O read the input matrices M, N
	float M[width][width], N[width][width], P[width][width];
	
	for (int i=0; i<width; i++){
		for(int j=0; j<width; j++){
			M[i][j] = 1;
			N[i][j] = 2;
		}
	}
	//M*N on the device
	float *Md, *Nd, *Pd;	
	int size = width*width*sizeof(float);
	// Load M and N to device mem
	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	//Allocate P on the device
	cudaMalloc((void**)&Pd, size);

	//Kernel invocation code
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid(width/TILE_WIDTH,width/TILE_WIDTH);
	sMatrixMulKernel<<<dimGrid,dimBlock>>>(Md, Nd, Pd, width);


	//Read P from the device
	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
	//Free device matrices
	cudaFree(Md); cudaFree(Nd); cudaFree(Pd);

	//I/O write the output matrix P
	for (int i=0; i<width; i++){
		for(int j=0; j<width; j++){
			printf("%f ", P[i][j]);
		}
		printf("\n");
	}
	
	//Free matrices M, N, P

	return 0;
}


__global__ void MatrixMulKernel(float *Md, float *Nd, float *Pd, int Width){

	//2D thread ID
	//int tx = threadIdx.x;
	//int ty = threadIdx.y;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	float Pvalue = 0;

	for(int k=0; k < Width; ++k){
		//float Mdelement = Md[ty * Width + k];
		//float Ndelement = Nd[k * Width + tx];
		//Pvalue += Mdelement * Ndelement;
		Pvalue += Md[Row * Width + k] * Nd[k * Width + Col];
	}

	Pd[Row * Width + Col] = Pvalue;
}


__global__ void sMatrixMulKernel(float *Md, float *Nd, float *Pd, int Width){

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	
	float Pvalue = 0;
	for(int m = 0; m < Width/TILE_WIDTH; ++m){
		Mds[ty][tx] = Md[Row*Width+(m*TILE_WIDTH+tx)];
		Nds[ty][tx] = Nd[Col+(m*TILE_WIDTH+ty)*Width];
		__syncthreads();
		for(int k=0; k < TILE_WIDTH; ++k){
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
	}
	Pd[Row*Width+Col] = Pvalue;
}
