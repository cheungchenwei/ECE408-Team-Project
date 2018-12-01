
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_
#define TILE_WIDTH 16

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

__constant__ float weight1[588];
__constant__ float weight2[14112];

__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
     

    int n = blockIdx.x * 2;
    // int n = blockIdx.x;  
    int m = blockIdx.y * 2;
    // int m = blockIdx.y;
    
    int h = ((blockIdx.z / W_grid) * TILE_WIDTH) + threadIdx.y;
    int w = ((blockIdx.z % W_grid) * TILE_WIDTH) + threadIdx.x;
    
    float acc = 0.0;
    float acc2 = 0.0;
    float acc3 = 0.0;
    float acc4 = 0.0;
	
	if (n < B && m < M && h < H_out && w < W_out){

        for( int c = 0; c < C; c++ ){
            for( int p = 0; p < K; p++ ){
                for( int q = 0; q < K; q++ ){
                    acc += x4d(n, c,  h + p, w + q) * k4d(m, c, p, q);

                    acc2 += x4d(n, c,  h + p, w + q) * k4d(m + 1, c, p, q);
                    acc3 += x4d(n + 1, c,  h + p, w + q) * k4d(m + 1, c, p, q);
                    acc4 += x4d(n + 1, c,  h + p, w + q) * k4d(m, c, p, q);

                }
            }
        }

        y4d(n, m, h, w) = acc;

        y4d(n, m+1, h, w) = acc2;
        y4d(n+1, m+1, h, w) = acc3;
        y4d(n+1, m, h, w) = acc4;

	}


#undef y4d
#undef x4d
#undef k4d
}

__global__ void forward_kernel_unroll1(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) weight1[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
     

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

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

// An example use of these macros:
// float a = y4d(0,0,0,0)
// y4d(0,0,0,0) = a
#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) weight2[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]


    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
     

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

__global__ void forward_kernel_shared(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

	const int H_out = H - K + 1;
    const int W_out = W - K + 1;
	int W_grid = ceil((W_out)/16.0);

	// An example use of these macros:
	// float a = y4d(0,0,0,0)
	// y4d(0,0,0,0) = a
	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

	int n, m, h, w, h0, w0, h_base, w_base;
	int X_tile_width = TILE_WIDTH + K-1;
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
    const int W_out = W - K + 1;
    int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
    int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
    const int Z = H_grid * W_grid;
    size_t shared_size = sizeof(float) * ((TILE_WIDTH + K-1) * (TILE_WIDTH + K-1) + K * K);

    // Set the kernel dimensions
    dim3 gridDim(B, M, Z);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);


    /* Optimization: Unroll + Weight matrix (kernel values) in constant memory
    if(C == 1){
        cudaMemcpyToSymbol(weight1, w.dptr_, M * C * K * K * sizeof(float));  
        // Call the kernel1
        forward_kernel_unroll1<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    } else{
        cudaMemcpyToSymbol(weight2, w.dptr_, M * C * K * K * sizeof(float));
        // Call the kernel2
        forward_kernel_unroll2<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, w.dptr_, B, M, C, H, W, K);
    }
    */

    // Optimization: Shared Memory convolution
    forward_kernel_shared<<<gridDim, blockDim, shared_size>>>(y.dptr_,x.dptr_,w.dptr_, B, M, C, H, W, K);


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