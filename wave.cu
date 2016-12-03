#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAXPOINTS 1000000
#define MAXSTEPS 1000000
#define MINPOINTS 20
#define PI 3.14159265
#define TILE_LENGTH 20

void check_param(void); //host
void init_line(void); //host
void printfinal (void); //host

int nsteps,                 	/* number of time steps */
    tpoints; 	     		/* total points along string */
float  values[MAXPOINTS+2], 	/* values at time t */
       oldval[MAXPOINTS+2], 	/* values at time (t-dt) */
       newval[MAXPOINTS+2]; 	/* values at time (t+dt) */

__global__ void updateKernel(float *values_d, float *oldval_d, float *newval_d, int Nsteps);

int main(int argc, char *argv[]){

	sscanf(argv[1],"%d",&tpoints);
	sscanf(argv[2],"%d",&nsteps);
	check_param();
	//Initialize the line
	printf("Initializing points on the line...\n");
	init_line();

	/***Update on the device***/
	printf("Updating all points for all time steps...\n");
	// values, oldval, newval on the device
	float *values_d, *oldval_d, *newval_d;
	int size = (MAXPOINTS+2)*sizeof(float);
	// load values, oldval to device mem
	cudaMalloc((void**)&values_d, size);
	cudaMemcpy(values_d, values, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&oldval_d, size);
	cudaMemcpy(oldval_d, oldval, size, cudaMemcpyHostToDevice);

	// allocate newval on the device mem
	cudaMalloc((void**)&newval_d, size);
	// kernel invocation code
	int numBlocks = ceil(tpoints / TILE_LENGTH); //if tpoints < TILE_LENGTH
	int threadsPerBlock = TILE_LENGTH; 
	updateKernel<<<numBlocks,threadsPerBlock>>>(values_d, oldval_d, newval_d, nsteps);

	//Read final values from the device
	cudaMemcpy(values, values_d, size, cudaMemcpyDeviceToHost);
	cudaFree(values_d); cudaFree(oldval_d); cudaFree(newval_d);

	printf("Printing final results...\n");
	printfinal();
	printf("\nDone.\n\n");
	
	return 0;
}
__global__ void updateKernel(float *values_d, float *oldval_d, float *newval_d, int Nsteps){

	int rank = blockIdx.x * blockDim.x + threadIdx.x;
	float dtime, c, dx, tau, sqtau;
	dtime = 0.3;
	c = 1.0;
	dx = 1.0;
	tau = (c * dtime / dx);
	sqtau = tau * tau;

	for(int i=1; i<=Nsteps; i++){
		newval_d[rank] = (2.0 * values_d[rank]) - oldval_d[rank] + (sqtau *  (-2.0)*values_d[rank]);
		oldval_d[rank] = values_d[rank];
		values_d[rank] = newval_d[rank];
	}
}


void check_param(void)
{
   char tchar[20];

   /* check number of points, number of iterations */
   while ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS)) {
      printf("Enter number of points along vibrating string [%d-%d]: "
           ,MINPOINTS, MAXPOINTS);
      scanf("%s", tchar);
      tpoints = atoi(tchar);
      if ((tpoints < MINPOINTS) || (tpoints > MAXPOINTS))
         printf("Invalid. Please enter value between %d and %d\n", 
                 MINPOINTS, MAXPOINTS);
   }
   while ((nsteps < 1) || (nsteps > MAXSTEPS)) {
      printf("Enter number of time steps [1-%d]: ", MAXSTEPS);
      scanf("%s", tchar);
      nsteps = atoi(tchar);
      if ((nsteps < 1) || (nsteps > MAXSTEPS))
         printf("Invalid. Please enter value between 1 and %d\n", MAXSTEPS);
   }

   printf("Using points = %d, steps = %d\n", tpoints, nsteps);

}

void init_line(void)
{
   int i, j;
   float x, fac, k, tmp;

   /* Calculate initial values based on sine curve */
   fac = 2.0 * PI;
   k = 0.0; 
   tmp = tpoints - 1;
   for (j = 1; j <= tpoints; j++) {
      x = k/tmp;
      values[j] = sin (fac * x);
      k = k + 1.0;
   } 

   /* Initialize old values array */
   for (i = 1; i <= tpoints; i++) 
      oldval[i] = values[i];
}

void printfinal(void)
{
   int i;

   for (i = 1; i <= tpoints; i++) {
      printf("%6.4f ", values[i]);
      if (i%10 == 0)
         printf("\n");
   }
}
