#include <stdio.h>
#include <algorithm>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

using namespace std;

#define block_size_x 16
#define block_size_y 16

int __host__ __device__ getIndex(const int i, const int j, const int width) {
    return i*width + j;
}

void GenArray(float** array, int Nx, int Ny, float Tedge, float Tcenter, float radius) {   
  

  float* array1 = new float[Nx*Ny];

  for (int i = 0; i < Nx; i++){ 
    for (int j = 0; j < Ny; j++){

      const int index = i*Ny + j;

      float distance2 = (i - Nx/2)*(i - Nx/2) + (j - Ny/2)*(j - Ny/2);

      if (distance2 < radius)
        array1[index] = Tcenter;
      else
        array1[index] = Tedge;
    }
  }

  *array = array1;
}

void Heat_transmition(float* Ahost, float* A, int Nx, int Ny, float alpha, float dx,float dy, float dt){
    float dx2 = dx*dx;
    float dy2 = dy*dy;

  for (int i = 1; i < Nx-1; i++){
    for (int j = 1; j < Ny-1; j++){
      const int index = getIndex(i,j,Ny);

      float u_ij = Ahost[index];
      float u_im1j = Ahost[getIndex(i-1, j, Ny)];
      float u_ip1j = Ahost[getIndex(i+1, j, Ny)];
      float u_ijm1 = Ahost[getIndex(i, j-1, Ny)];
      float u_ijp1 = Ahost[getIndex(i, j+1, Ny)];

      A[index] = u_ij + alpha * dt * ((u_im1j - 2.0*u_ij + u_ip1j)/dx2 + (u_ijm1 - 2.0*u_ij + u_ijp1)/dy2);
    }
  }
}

void Write(float* arr, int Nx, int Ny, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    for (int i = 0; i < Nx*Ny; i++){
        fprintf(fp, "%f ", arr[i]);
        if (i % Nx == (Nx-1))
              fprintf(fp,"\n");
    }
    fclose(fp);
}

/*
 *  Kernel 1
 */
__global__ void KernelGPU(const float* array, float* arrayp1, const int Nx, const int Ny, const float dx2, const float dy2, const float alpha, const float dt){

  __shared__ float s_Array[(block_size_x + 2)*(block_size_y + 2)];
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;

  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;
  int s_ny = block_size_y + 2;

  // Cuadrado central
  s_Array[getIndex(s_i,s_j,s_ny)] = array[getIndex(i,j,Ny)];

  // Borde superior
  if (s_i == 1 && s_j != 1 && i != 0){
    s_Array[getIndex(0, s_j, s_ny)] = array[getIndex(blockIdx.x*blockDim.x - 1, j, Ny)];
  }

  // Borde inferior
  if (s_i == block_size_x && s_j != block_size_y + 1  && i != Nx -1){
    s_Array[getIndex(block_size_x + 1, s_j, s_ny)] = array[getIndex((blockIdx.x + 1)*blockDim.x, j, Ny)];
  }

  // Borde izquierdo
  if (s_i != 1 && s_j==1 && j != 0){
    s_Array[getIndex(s_i, 0, s_ny)] = array[getIndex(i, blockIdx.y*blockDim.y -1, Ny)]; 
  }

  // Borde derecho
  if (s_i != block_size_x && s_j == block_size_y && j != Ny-1){
    s_Array[getIndex(s_i, block_size_y+1, s_ny)] = array[getIndex(i, (blockIdx.y + 1)*blockDim.y, Ny)];
  }

  __syncthreads();

  if (i > 0 && i < Nx - 1){
    if (j > 0 && j < Ny - 1){
      float u_ij   = s_Array[getIndex(s_i, s_j, s_ny)];
      float u_im1j = s_Array[getIndex(s_i - 1, s_j, s_ny)];
      float u_ip1j = s_Array[getIndex(s_i + 1, s_j, s_ny)];
      float u_ijm1 = s_Array[getIndex(s_i, s_j - 1, s_ny)];
      float u_ijp1 = s_Array[getIndex(s_i, s_j + 1, s_ny)];

      // float u_ij   = s_Array[getIndex(i,j,Ny)];
      // float u_im1j = s_Array[getIndex(i-1, j, Ny)];
      // float u_ip1j = s_Array[getIndex(i+1, j, Ny)];
      // float u_ijm1 = s_Array[getIndex(i, j-1, Ny)];
      // float u_ijp1 = s_Array[getIndex(i, j+1, Ny)];

      arrayp1[getIndex(i, j, Ny)] = u_ij + alpha * dt * ( (u_im1j - 2.0*u_ij + u_ip1j)/dx2 + (u_ijm1 - 2.0*u_ij + u_ijp1)/dy2 );
    }
  }
}

/*
 *  Kernel 2 (Shared Memory)
 */
// __global__ void KernelGPU2(const float* array, float* arrayp1, const int Nx, const int Ny, const float dx2, const float dy2, const float alpha, const float dt){

//   __shared__ float s_Array[block_size_x + 2][block_size_y +2];

//   //Filas y columnas del bloque
//   int row = threadIdx.x + blockIdx.x*blockDim.x;
//   int col = threadIdx.y + blockIdx.y*blockDim.y;

//   //Fila y columna de la memoria compartida
//   int i = threadIdx.x + 1; // sin contar el borde superior 
//   int j = threadIdx.y + 1; // sin contar el borde izquierdo 
//   int ny = block_size_y +2; // Cantidad de columnas del sub-placa

//   //Rellenar memoria compartida
//   //Cuadro central
//   s_Array[getIndex(i,j,ny)] = array[getIndex(row,col,ny)];







// }


/*
 *  Codigo Principal
 */
int main(int argc, char **argv){
/*
 *  Inicializacion
 */
//------------- variables de tiempo -----------------
  clock_t t1, t2;
  double ms;
  cudaEvent_t ct1, ct2;
  float d_time;

//--------------- variables de CPU -------------------
  const int Nx = 200; // 10 metros, largo de la barra
  const int Ny = 200; // 10 metros, ancho de la barra

  const float alpha = 0.5; // depende del material

  const float dx = 0.01; // [m] = 1 [cm] -> 1000 [cm] = 10 [m]
  const float dy = 0.01;

	const float dx2 = dx*dx;
	const float dy2 = dy*dy;

  const float dt = dx2*dy2 / (2.0 * alpha * (dx2 + dy2)); // delta tiempo
  // float dt = 0.00001;

  const int t = 1000; // cantidad de iteraciones
  const int outputEveryTime = 200;

  float Tedge = 25.0;
  float Tcenter = 200.0;
  // float radius = 5;
  float radius = (Nx/6.0) * (Nx/6.0); //m 

  printf("Radio: %f", radius);

  int nElements = Nx*Ny;

  float *Ahost;
  float *AhostGPU;
  float* Ahostp1 = (float*)calloc(nElements, sizeof(float));

  GenArray(&Ahost, Nx, Ny, Tedge, Tcenter, radius);
  GenArray(&AhostGPU, Nx, Ny, Tedge, Tcenter, radius);

  printf("\n");
  printf("\n");

  memcpy(Ahostp1, Ahost, nElements*sizeof(float));

//--------------- variables de GPU ------------------- 

//----------------------- CPU ------------------------
  t1 = clock();
  
  /*for (int n = 0; n <= t; n++){
    
    Heat_transmition(Ahost, Ahostp1, Nx, Ny, alpha, dx, dy, dt);
    
    // Printea el arreglo cada 20 iteraciones.
		if (n % outputEveryTime == 0){
      int i = n/outputEveryTime;
      char filename[64];
      sprintf(filename, "archivos/transmition_CPU_%03d.txt", i);
      Write(Ahost, Nx, Ny, filename);
		}
    swap(Ahost, Ahostp1);

  }*/

  t2 = clock();
  ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
  printf("\nTiempo CPU: %f[ms]\n", ms);

//-------------------- kernel 1 -----------------------
//---------------- variables de GPU -------------------


  float* Bhost;
  float* Bhostp1;

  cudaMalloc((void**)&Bhost, nElements*sizeof(float));
  cudaMalloc((void**)&Bhostp1, nElements*sizeof(float));

  cudaMemcpy(Bhost, AhostGPU, nElements*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(Bhostp1, AhostGPU, nElements*sizeof(float), cudaMemcpyHostToDevice);

  dim3 gs(Nx/block_size_x + 1, Ny/block_size_y +1); //gs
  dim3 bs(block_size_x, block_size_y); // bs

  printf("GS: {%d %d %d}\n", gs.x, gs.y, gs.z);
  printf("BS: {%d %d %d}\n", bs.x, bs.y, bs.z);

  // printf("GS: %d\n", gs);

  // float* Ahostgpu = new float[nElements];

  cudaEventCreate(&ct1);
  cudaEventCreate(&ct2);
  cudaEventRecord(ct1);
  for (int n = 0; n <= t; n++){

    KernelGPU<<<gs, bs>>>(Bhost, Bhostp1, Nx, Ny, dx2, dy2, alpha, dt);
    // Printea el arreglo cada 20 iteraciones.
		if (n % outputEveryTime == 0){
      cudaMemcpy(AhostGPU, Bhost, nElements*sizeof(float), cudaMemcpyDeviceToHost);

      int i = n/outputEveryTime;
      char filename[64];
      sprintf(filename, "archivos/transmition_GPU_%03d.txt", i);
      Write(AhostGPU, Nx, Ny, filename);
		}

    std::swap(Bhost, Bhostp1);
  }

  cudaEventRecord(ct2);
  cudaEventSynchronize(ct2);
  cudaEventElapsedTime(&d_time, ct1, ct2);
  printf("Tiempo para GPU: %f [ms]\n", d_time);
  
	cudaFree(Bhost); cudaFree(Bhostp1); 
  delete[] Ahost; delete[] Ahostp1; 
  free(AhostGPU);

  return 0;
}
