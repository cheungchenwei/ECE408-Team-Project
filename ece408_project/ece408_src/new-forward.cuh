
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 32

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float weight1[588];
__constant__ float weight2[14112];

__global__ void forward_kernel_unroll1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = 66;
    const int W_out = 66;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) weight1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int W_grid = 3;
     

    //int n = blockIdx.x * 2;
     int n = blockIdx.x;  
    //int m = blockIdx.y * 2;
     int m = blockIdx.y;
    
    int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
    int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;
    
    float acc = 0.0;

	
	if (n < B && m < M && h < H_out && w < W_out){

            /*
            Optimization: convolution loop unrolling, using "#pragma unroll"
            Op Time: 0.053147
            Op Time: 0.133266
            Reason: Since the kernel size is fixed(7), we can use loop unrolling to transfer dynamic loop into static loop.
            Loop unrolling is effective due to the memory access in computer.
            For example, "for(int i = 0; i < 2; i++) b[i] += 1" is slower than "b[0] += 1; b[1] += 1;". This is because in 
            the former version, variable i increases 2 times and assign to b 2 times, whereas the latter version only increase i by 2.
            */
            
        for( int c = 0; c < C; c++){
            #pragma unroll 7
            for( int p = 0; p < K; p++){
                #pragma unroll 7
                for( int q = 0; q < K; q++){
                    acc += x4d(n, c,  h + p, w + q) * k4d(m, c, p, q);
                
                }
            }
        }
            
        

        y4d(n, m, h, w) = acc;


	}


#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_unroll2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = 27;
    const int W_out = 27;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) weight2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int W_grid = 1;
     

    //int n = blockIdx.x * 2;
     int n = blockIdx.x;  
    //int m = blockIdx.y * 2;
     int m = blockIdx.y;
    
    int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
    int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;
    
    float acc = 0.0;

	
	if (n < B && m < M && h < H_out && w < W_out){

            /*
            Optimization: convolution loop unrolling, using "#pragma unroll"
            Op Time: 0.053147
            Op Time: 0.133266
            Reason: Since the kernel size is fixed(7), we can use loop unrolling to transfer dynamic loop into static loop.
            Loop unrolling is effective due to the memory access in computer.
            For example, "for(int i = 0; i < 2; i++) b[i] += 1" is slower than "b[0] += 1; b[1] += 1;". This is because in 
            the former version, variable i increases 2 times and assign to b 2 times, whereas the latter version only increase i by 2.
            */
            
        for( int c = 0; c < C; c++){
            #pragma unroll 7
            for( int p = 0; p < K; p++){
                #pragma unroll 7
                for( int q = 0; q < K; q++){
                    acc += x4d(n, c,  h + p, w + q) * k4d(m, c, p, q);
                
                }
            }
        }
            
        

        y4d(n, m, h, w) = acc;


	}


#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_shared1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	int W_grid = ceil((W_out)/8.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) weight1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int n, m, h, w, h0, w0, h_base, w_base;
	int X_tile_width = TILE_WIDTH + K - 1;
	extern __shared__ float shared_mem[];
	float* X_shared = &shared_mem[0];
	float* W_shared = &shared_mem[X_tile_width * X_tile_width];
	n = blockIdx.x;
	m = blockIdx.y;
	h0 = threadIdx.y;
	w0 = threadIdx.x;
	h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
	w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
	h = h_base + h0;
	w = w_base + w0;
	float acc = 0.0;
	for(int c = 0; c < C; c++) {	
        // sum over all input channels
		//load weights for W [m, c,..],
		// h0 and w0 used as shorthand for threadIdx.x
		// and threadIdx.y
		if ((h0 < K) && (w0 < K)) {
			W_shared[h0 * K + w0]= k4d(m, c, h0, w0);
		}
		__syncthreads();

		// load tile from X[n, c,…] into shared memory
		for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
			for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
				if (i < H && j < W) {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(n, c, i, j);
				} else {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0;
				}
			}
		}
		__syncthreads();

		for(int p = 0; p < K; p++) {
			for(int q = 0; q < K; q++) {
				if((h0 + p) < X_tile_width && (w0 + q) < X_tile_width) {
					acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
				}
			}
		}
		__syncthreads();
	}

	if (n < B && m < M && h < H_out && w < W_out) {
		y4d(n, m, h, w) = acc;
    }
    
	#undef y4d
	#undef x4d
	#undef k4d
}

__global__ void forward_kernel_shared2(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	int W_grid = ceil((W_out)/8.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) weight2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int n, m, h, w, h0, w0, h_base, w_base;
	int X_tile_width = TILE_WIDTH + K - 1;
	extern __shared__ float shared_mem[];
	float* X_shared = &shared_mem[0];
	float* W_shared = &shared_mem[X_tile_width * X_tile_width];
	n = blockIdx.x;
	m = blockIdx.y;
	h0 = threadIdx.y;
	w0 = threadIdx.x;
	h_base = (blockIdx.z / W_grid) * TILE_WIDTH; // vertical base out data index for the block
	w_base = (blockIdx.z % W_grid) * TILE_WIDTH; // horizontal base out data index for the block
	h = h_base + h0;
	w = w_base + w0;
    float acc = 0.0;
    /*
            Optimization: Shared Memory convolution
            Reason: For this optimization, we load tiles from X[n, c,…] into shared memory which could be reused for multiple times when do the convolution with W.
            Thus, the kernel could perform more efficient since it reduces the time that costs to read from global memory and write back to global memory. 
    */
	for(int c = 0; c < C; c++) {	

		if ((h0 < K) && (w0 < K)) {
			W_shared[h0 * K + w0]= k4d(m, c, h0, w0);
		}
		__syncthreads();

		for (int i = h; i < h_base + X_tile_width; i += TILE_WIDTH) {
			for (int j = w; j < w_base + X_tile_width; j += TILE_WIDTH) {
				if (i < H && j < W) {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = x4d(n, c, i, j);
				} else {
					X_shared[(i - h_base) * X_tile_width + (j - w_base)] = 0;
				}
			}
		}
		__syncthreads();

		for(int p = 0; p < K; p++) {
			for(int q = 0; q < K; q++) {
				if((h0 + p) < X_tile_width && (w0 + q) < X_tile_width) {
					acc += X_shared[(h0 + p) * X_tile_width + (w0 + q)] * W_shared[p * K + q];
				}
			}
		}
		__syncthreads();
	}

	if (n < B && m < M && h < H_out && w < W_out) {
		y4d(n, m, h, w) = acc;
    }
    
	#undef y4d
	#undef x4d
	#undef k4d
}


__global__ void forward_kernel_atomic(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{
    const int H_out = 27;
    const int W_out = 27;

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) weight2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int W_grid = 1;
    int H_grid = 1;
    
    int n = blockIdx.x;  
    int m = blockIdx.y;
    
    int h = ((blockIdx.z / C / W_grid) * TILE_WIDTH) + threadIdx.y;
    int w = ((blockIdx.z / C % W_grid) * TILE_WIDTH) + threadIdx.x;
    int c = (blockIdx.z / (H_grid*W_grid));
    /*
        Optimization: 
        Op Time:0.112061 (Only for C > 1)
        Reason: Although we treat C in parallel, this optimization is slower due to atomic operation works as serial functions.
        Therefore, atomic operation is better for communication betweern threads in different blocks, but not good for our performance in this project.
    */

    float acc = 0.0;

	if (n < B && m < M && h < H_out && w < W_out){
            
        #pragma unroll 7
        for( int p = 0; p < K; p++){
            #pragma unroll 7
            for( int q = 0; q < K; q++){
                acc += x4d(n, c,  h + p, w + q) * k4d(m, c, p, q);
                
            }
        }
        atomicAdd(&y4d(n, m, h, w),acc);
        //y4d(n, m, h, w) = acc;
	}

#undef y4d
#undef x4d
#undef k4d
}

__global__ void sharedMatrixMultiply(float *A, float *B, float *C,
    int numARows, int numAColumns,
    int numBRows, int numBColumns,
    int numCRows, int numCColumns) {

    int TILEWIDTH = 16;
    __shared__ float tileA[TILEWIDTH][TILEWIDTH];
    __shared__ float tileB[TILEWIDTH][TILEWIDTH];

    int col = threadIdx.x + blockIdx.x * TILEWIDTH;
    int row = threadIdx.y + blockIdx.y * TILEWIDTH;

    /*
            Optimization: Shared Memory Matrix Multiply
            Reason: For this optimization, we optimized matrix multiply using shared memory, which could increase re-use and avoid thread divergence.
            Besides, in the matrix multiply, we used coalescing techniques, which can more effectively move data from the global memory into shared memories and registers, allowing the DRAMs to supply data at high rate.
            Thus, the optimization could allow CUDA device to utilize the global memory bandwidth more efficiently.
    */

    float val = 0;

    for(int i = 0; i < ceil(1.0 * numAColumns / TILEWIDTH); i++) {
        if(row < numARows && (i * TILEWIDTH + threadIdx.x) < numAColumns) {
            tileA[threadIdx.y][threadIdx.x] = A[row * numAColumns + (i * TILEWIDTH + threadIdx.x)];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0;
        }

        if(col < numBColumns && (i * TILEWIDTH + threadIdx.y) < numBRows) {
            tileB[threadIdx.y][threadIdx.x] = B[(i * TILEWIDTH + threadIdx.y) * numBColumns + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for(int j = 0; j < TILEWIDTH; j++) {
            val += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }

        __syncthreads();
    }

    if(row < numCRows && col < numCColumns) {
        C[row * numCColumns + col] = val;
    }
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];
    const int H_out = H - K + 1;
    const int W_out = H - K + 1;
    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
    const int Z = H_grid * W_grid;
    //size_t shared_size = sizeof(float) * ((TILE_WIDTH + K-1) * (TILE_WIDTH + K-1) + K * K);

    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Optimization: Unroll + Weight matrix (kernel values) in constant memory
    if(C == 1){
        // kernel for layer 1, which has smaller size
        // put weight matrix in constant memory
        cudaMemcpyToSymbol(weight1, w.dptr_, 588 * sizeof(float));  
        // Call the kernel1
        forward_kernel_unroll1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    } else{
        // kernel for layer 2, which has greater size
        // put weight matrix in constant memory
        cudaMemcpyToSymbol(weight2, w.dptr_, 14112 * sizeof(float));
        // Call the kernel2
        forward_kernel_unroll2<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    }
    
    

    /* Optimization: Shared Memory convolution
    if(C == 1){
        // kernel for layer 1, which has smaller size
        // put weight matrix in constant memory
        cudaMemcpyToSymbol(weight1, w.dptr_, 588 * sizeof(float));  
        // Call the kernel1
        forward_kernel_shared1<<<gridDim, blockDim, shared_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
    } else{
        // kernel for layer 2, which has larger size
        // put weight matrix in constant memory
        cudaMemcpyToSymbol(weight2, w.dptr_, 14112 * sizeof(float));
        // Call the kernel2
        forward_kernel_shared2<<<gridDim, blockDim, shared_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
    }
    */
    
    /* Optimization: Atomic reduction
    // we need to add C into gridDim, then every thread is able to use atomic function. 
    dim3 gridDimAtomic(B, M, Z * C);;
    dim3 blockDimAtomic(TILE_WIDTH, TILE_WIDTH, 1);
    if(C == 1){
        // kernel for layer 1, which has smaller size
        // put weight matrix in constant memory
        cudaMemcpyToSymbol(weight1, w.dptr_, 588 * sizeof(float));  
        // Call the kernel1
        forward_kernel_shared1<<<gridDim, blockDim, shared_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);
    } else{
        // kernel for layer 2, which has larger size
        // put weight matrix in constant memory

        cudaMemcpyToSymbol(weight2, w.dptr_, 14112 * sizeof(float));
        // Call the kernel2
        forward_kernel_atomic<<<gridDimAtomic, blockDimAtomic>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    
    }
    */

    /* Optimization: Shared Memory Matrix Multiply
    int W_unroll = H_out * W_out;
	int H_unroll = C * K * K;
	float* X_unrolled;
    cudaMalloc((void **) &X_unrolled, W_unroll * H_unroll * sizeof(float));
    TILEWIDTH = 16;
    NUM_THREADS = 1024;
	dim3 dimBlock(TILEWIDTH, TILEWIDTH, 1);
	dim3 dimGrid(ceil((1.0 * W_unroll)/TILEWIDTH), ceil((1.0 * M)/TILEWIDTH), 1);
	int num_blocks = ceil((1.0 * C * H_out * W_out) / NUM_THREADS);
	for (int b = 0; b < B; b++) {
		float* output = &y.dptr_[b * M * H_out * W_out];
        sharedMatrixMultiply<<<dimGrid, dimBlock>>>(w.dptr_, X_unrolled, output, M, H_unroll, H_unroll, W_unroll, M, W_unroll);
    }
    */

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif