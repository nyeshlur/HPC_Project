#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda/common.hpp>

/* Compile: nvcc -std=c++11 Cuda_implementation.cu -o Cuda -I/usr/local/include/opencv4 -lopencv_core  -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc  $(pkg-config opencv4 --libs)
   Run: ./Cuda
   Profiler commands:
   nv-nsight-cu-cli ./Cuda
   nsys profile --stats=true --force-overwrite true --show-output true ./Cuda */

#define BLOCK_SIZE 16

using namespace std;
using namespace cv;

/********** Calculating Summation in Point Spread Function (PSF) ***********************/
__global__ void calcPSF(cuda::PtrStepSz<float> outputImg, Size filterSize, int radius,
int * summa, cuda::PtrStepSz<float> summa_mat) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int size = filterSize.height;
    int midx = size/2;
    int midy = size/2;

    if(row<size && col<size){
        outputImg(row, col) = 0.0;
    }
   __syncthreads();

    if(row <= radius+radius && row >= 0 && col <= radius+radius && col >= 0) {
        if((row-radius) * (row-radius) + (col-radius) * (col-radius) <= radius*radius) {
            outputImg(midx+row-radius, midy+col-radius) = 255.0;
            atomicAdd(&summa[0], 255.0);
        }
    }
    __syncthreads();

    if(row <= radius+radius && row >= 0 && col <= radius+radius && col >= 0) {
        if((row-radius) * (row-radius) + (col-radius) * (col-radius) <= radius*radius) {
            summa_mat(0,0) = summa[0];
        }
    }
    __syncthreads();
}

/********** Normalizing Point Spread Function (PSF) **************************/
__global__ void psf_normalize(int summation, cuda::PtrStepSz<float> outputImg){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int size = outputImg.rows;
    if(summation!=0 && row<size && col<size && outputImg(row,col)!=0.0f){
        outputImg(row,col) = outputImg(row,col)/summation;
    }
    __syncthreads();
}

/********** FFT Shift of Point Spread Function **********************************/
__global__ void fft_shift(cuda::PtrStepSz<float> input_PSF,
cuda::PtrStepSz<float> output_PSF, int N) {

    int sLine = N;
    int sSlice = N * N;

    // Transformations Equations
    int sEq1 = (sSlice + sLine) / 2;
    int sEq2 = (sSlice - sLine) / 2;

    __syncthreads();

    // Thread Index (1D)
    int xThreadIdx = threadIdx.x;
    int yThreadIdx = threadIdx.y;

    __syncthreads();

    // Block Width & Height
    int blockWidth = blockDim.x;
    int blockHeight = blockDim.y;

    __syncthreads();

    // Thread Index (2D)
    int xIndex = blockIdx.x * blockWidth + xThreadIdx;
    int yIndex = blockIdx.y * blockHeight + yThreadIdx;

    __syncthreads();

    // Thread Index Converted into 1D Index
    int index = (yIndex * N) + xIndex;
    __syncthreads();

    if (xIndex < N / 2){
        if (yIndex < N / 2) {
             output_PSF(0,index) = input_PSF(0,index + sEq1);
             __syncthreads();
        }
        else {
            output_PSF(0,index) = input_PSF(0,index - sEq2);
            __syncthreads();
        }
    }
    else {
        if (yIndex < N / 2) {
            output_PSF(0,index) = input_PSF(0,index + sEq2);
            __syncthreads();
        }
        else{
            output_PSF(0,index) = input_PSF(0,index - sEq1);
            __syncthreads();
        }
    }

}

/********** Merge Image/Filter with zero values plane - Complex data type **********/
template< typename T_in,typename T_out>
__global__ void mergefilter(cuda::PtrStepSz<T_in> input,
cuda::PtrStepSz<T_out> output) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input.rows*input.cols) && j < (input.rows*input.cols)){
        output(i,j).x =  input(i,j);
        output(i,j).y =  0.0f;
        __syncthreads();

    }
}

/********** Split Image/Filter containing two planes to a single plane ************/
__global__ void splitfilter(cuda::PtrStepSz<float> input,
cuda::PtrStepSz<float> output) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input.rows*input.cols) && j < (input.rows*input.cols)){
        output(i,j) = input(i,j+j);
    }
}

/********** Adding and Division as part of restoration process ****************/
__global__ void pow_add_div_filter(cuda::PtrStepSz<float> input,
cuda::PtrStepSz<float> output, double nsr) {

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input.rows*input.cols) && j < (input.rows*input.cols)){
        output(i,j) = nsr + (abs(input(i,j))*abs(input(i,j)));
        output(i,j) = input(i,j)/output(i,j);
    }
}

/********** Dot product of image and wiener filter ****************************/
__global__ void mulSpectrums(cuda::PtrStepSz<float2> complexI,
cuda::PtrStepSz<float2> complexH, cuda::PtrStepSz<float2> complexIH) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (complexI.rows*complexI.cols) && j < (complexI.rows*complexI.cols)){
        float I_real = complexI(i,j).x;
        float I_img = complexI(i,j).y;
        float H_real = complexH(i,j).x;
        float mul_real = I_real * H_real;
        float mul_imag = I_img * H_real;
        complexIH(i,j).x =  mul_real;
        complexIH(i,j).y =  mul_imag;
    }
}

/********** Normalize final restored image - Fits into (0-255) range based on min and max ********************/
__global__ void normalize_img(cuda::PtrStepSz<uint8_t> input,
cuda::PtrStepSz<uint8_t> output, uint8_t min, uint8_t max) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < (input.rows*input.cols) && j < (input.rows*input.cols)){
        uint8_t temp = round(((uint8_t)input(i, j) - min) * (255.0/(max-min)));
        output(i, j) = temp;
    }
    __syncthreads();
}

/********** Calculate correctness of algorithm using MSE and PSNR ****************************/
/* PSNR - Peak Signal to noise Ratio
   MSE - Mean Squared Error */
double getPSNR(const Mat& I1, const Mat& I2, int R, int snr, double *mse, double *psnr)
{
    Mat s1;
    absdiff(I1, I2, s1); // |I1 - I2|
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);   // |I1 - I2|^2
    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];

    if( sse <= 1e-10)
        return 0;
    else
    {
        *mse =sse /(double)(I1.channels() * I1.total());
        *psnr = 10.0*log10((255*255)/(*mse));
    }
    return 0;
}


int main()
{
    std::chrono::time_point<std::chrono::system_clock> startPSF, endPSF, startNormalize, endNormalize, startfft, endfft, startmerge, endmerge, startimgnorm, endimgnorm, startsplit1, endsplit1, startsplit2, endsplit2, startpadf, endpadf, startmerge2, endmerge2, startmerge3, endmerge3, startmulspec, endmulspec, startdft, enddft, startidft, endidft;

    int R = 2; // Radius of PSF function
    int snr = 105;
    double psnr = 0.0f;
    double mse  = 0.0f;

    Mat imgIn;
    imgIn = imread("final_images/bear_256.png", IMREAD_GRAYSCALE);
    int width = imgIn.size().width;
    int height = imgIn.size().height;

    // Making sure the images are of square size and even sized
    if(width > height)
        if(height % 2 == 0)
            imgIn = imgIn(Range(0,height), Range(0,height));
        else
            imgIn = imgIn(Range(0,height-1), Range(0,height-1));
    else if(height > width)
        if(width % 2 == 0)
            imgIn = imgIn(Range(0,width), Range(0,width));
        else
            imgIn = imgIn(Range(0,width-1), Range(0,width-1));
    Mat blurIn = imgIn.clone();

    /**************** Start - Blur image *******************************************/
    GaussianBlur(imgIn, blurIn, Size(5,5), 0);
    imwrite("final_images/blur_cuda_256.jpg", blurIn);
    /**************** End - Blur image **********************************************/


    /**************** Start - Calculate PSF ********************************************/
    int *d_summa;
    int *summa;
    int summa_size = 1 * 1 * sizeof( int);
    cudaMalloc((void **)&d_summa, summa_size);
    summa = ( int *)malloc(summa_size);
    summa[0] = 0;

    Mat imgOut;
    Rect roi = Rect(0, 0, blurIn.cols & -2, blurIn.rows & -2);

    Mat Hw, h(roi.height, roi.width, CV_32FC1),h_host(roi.height, roi.width, CV_32FC1);

    cuda::GpuMat h_cuda, h_shifted(roi.height, roi.width, CV_32FC1);
    h_cuda.upload(h);

    cudaMemcpy(d_summa, summa, summa_size, cudaMemcpyHostToDevice);

    dim3 dimgrid3(roi.height/BLOCK_SIZE,roi.width/BLOCK_SIZE);
    dim3 dimblock3(BLOCK_SIZE,BLOCK_SIZE);

    cuda::GpuMat summa_cuda(1, 1, CV_32FC1);
    Mat summa_host(1, 1, CV_32FC1);
//
    startPSF = std::chrono::system_clock::now();
    calcPSF<<<dimgrid3,dimblock3>>>(h_cuda, roi.size(), R, d_summa, summa_cuda);
    cudaDeviceSynchronize();
    endPSF = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedtime = endPSF - startPSF;
//
    h_cuda.download(h);
    summa_cuda.download(summa_host);
    int summation = summa_host.at<float>(0,0);
//
    startNormalize = std::chrono::system_clock::now();
    psf_normalize<<<dimgrid3,dimblock3>>>(summation, h_cuda);
    cudaDeviceSynchronize();
    endNormalize = std::chrono::system_clock::now();
    elapsedtime += endNormalize - startNormalize;
//
    h_cuda.download(h);
    /**************** End - Calculate PSF ********************************************/


    /**************** Start - Wiener Filter ******************************************/
    int cx = h.cols / 2;
    int cy = h.rows / 2;
    cuda::GpuMat fftshift_input_cuda;
    cuda::GpuMat fftshift_output_cuda(roi.height*roi.width,1,CV_32FC1);

    cuda::GpuMat merge_output_cuda(roi.height, roi.width, CV_32FC2);
    cuda::GpuMat dft_output_cuda(roi.height, roi.width, CV_32FC2);
    cuda::GpuMat split_output_cuda(roi.height, roi.width, CV_32FC1);
    cuda::GpuMat pow_add_div_output_cuda(roi.height, roi.width, CV_32FC1);

    int threads = cx * cy;
    Mat h_flat = h.reshape(1,roi.height*roi.width);
    cuda::GpuMat h_flat_cuda(roi.height*roi.width, 1, CV_32FC1);
    h_flat_cuda.upload(h_flat);
//
    startfft = std::chrono::system_clock::now();
    fft_shift<<<dimgrid3,dimblock3>>>(h_flat_cuda, fftshift_output_cuda, h.cols);
    cudaDeviceSynchronize();
    endfft = std::chrono::system_clock::now();
    elapsedtime += endfft - startfft;
//
    Mat fftshift_output_host(roi.height*roi.width,1,CV_32FC1);
    fftshift_output_cuda.download(fftshift_output_host);
    fftshift_output_host = fftshift_output_host.reshape (1, roi.width);
    fftshift_output_cuda = fftshift_output_cuda.reshape (1, roi.width);
//
    startmerge = std::chrono::system_clock::now();
    mergefilter<float, float2><<<dimgrid3,dimblock3>>>(fftshift_output_cuda, merge_output_cuda);
    cudaDeviceSynchronize();
    endmerge = std::chrono::system_clock::now();
    elapsedtime += endmerge - startmerge;
//
    Mat merge_output_host(h.rows,h.cols,CV_32FC2);
    merge_output_cuda.download(merge_output_host);

    Mat dft_output_host(h.rows,h.cols,CV_32FC2);
    dft(merge_output_host, dft_output_host);

    dft_output_cuda.upload(dft_output_host);
//
    startsplit1 = std::chrono::system_clock::now();
    splitfilter<<<dimgrid3,dimblock3>>>(dft_output_cuda, split_output_cuda);
    cudaDeviceSynchronize();
    endsplit1 = std::chrono::system_clock::now();
    elapsedtime += endsplit1 - startsplit1;
//
//
    startpadf = std::chrono::system_clock::now();
    pow_add_div_filter<<<dimgrid3,dimblock3>>>(split_output_cuda, pow_add_div_output_cuda, 1.0 / double(snr));
    cudaDeviceSynchronize();
    endpadf = std::chrono::system_clock::now();
    elapsedtime += endpadf - startpadf;
//

    Mat pow_add_div_output_host(h.rows,h.cols,CV_32FC1);
    pow_add_div_output_cuda.download(pow_add_div_output_host);

    h_flat.release(); fftshift_output_host.release();
    /**************** End - Wiener Filter ********************************************/

    /**************** Start - Image Restoration using Filter *****************************************/
    cuda::GpuMat blurIn_cuda(roi.height, roi.width, CV_32FC1);
    cuda::GpuMat complexI(roi.height, roi.width, CV_32FC2);
    cuda::GpuMat complexH(roi.height, roi.width, CV_32FC2);
    cuda::GpuMat complexIH_split_cuda(roi.height, roi.width, CV_32FC1);
    cuda::GpuMat imgOut_cuda(roi.height, roi.width, CV_32FC1);
    cuda::GpuMat imgOut_norm_cuda(roi.height, roi.width, CV_8U);

    Mat complexI_host(roi.height, roi.width,CV_32FC2);
    Mat complexH_host(roi.height, roi.width,CV_32FC2);
    Mat complexIH_split_host(roi.height, roi.width,CV_32FC1);
    Mat imgOut_norm_host(roi.height, roi.width,CV_8U);

    blurIn_cuda.upload(blurIn);
//
    startmerge2 = std::chrono::system_clock::now();
    mergefilter<uint8_t, float2><<<dimgrid3,dimblock3>>>(blurIn_cuda, complexI);
    cudaDeviceSynchronize();
    endmerge2 = std::chrono::system_clock::now();
    elapsedtime += endmerge2 - startmerge2;
//
    complexI.download(complexI_host);
    startdft = std::chrono::system_clock::now();
    dft(complexI_host, complexI_host, DFT_SCALE);
    enddft = std::chrono::system_clock::now();
    elapsedtime += enddft - startdft;
    complexI.upload(complexI_host);
//
    startmerge3 = std::chrono::system_clock::now();
    mergefilter<float, float2><<<dimgrid3,dimblock3>>>(pow_add_div_output_cuda, complexH);
    cudaDeviceSynchronize();
    endmerge3 = std::chrono::system_clock::now();
    elapsedtime += endmerge3 - startmerge3;
//
    complexH.download(complexH_host);

    cuda::GpuMat complexIH(roi.height, roi.width, CV_32FC2);
    Mat complexIH_host(roi.height, roi.width,CV_32FC2);
//
    startmulspec = std::chrono::system_clock::now();
    mulSpectrums<<<dimgrid3,dimblock3>>>(complexI, complexH, complexIH);
    cudaDeviceSynchronize();
    endmulspec = std::chrono::system_clock::now();
    elapsedtime += endmulspec - startmulspec;
//
    complexIH.download(complexIH_host);
    startidft = std::chrono::system_clock::now();
    idft(complexIH_host, complexIH_host);
    endidft = std::chrono::system_clock::now();
    elapsedtime += endidft - startidft;
    complexIH.upload(complexIH_host);
//
    startsplit2 = std::chrono::system_clock::now();
    splitfilter<<<dimgrid3,dimblock3>>>(complexIH, complexIH_split_cuda);
    cudaDeviceSynchronize();
    endsplit2 = std::chrono::system_clock::now();
    elapsedtime += endsplit2 - startsplit2;
//
    complexIH_split_cuda.download(complexIH_split_host);
    /**************** End - Image Restoration using Filter ********************************************/


    /**************** Start - Normalization ********************************************/
    complexIH_split_host.convertTo(imgOut, CV_8U);
    imgOut_cuda.upload(imgOut);
    double min, max;
    minMaxLoc(imgOut, &min, &max);
//
    startimgnorm = std::chrono::system_clock::now();
    normalize_img<<<dimgrid3,dimblock3>>>(imgOut_cuda, imgOut_norm_cuda, min, max);
    cudaDeviceSynchronize();
    endimgnorm = std::chrono::system_clock::now();
    elapsedtime += endimgnorm - startimgnorm;
//
    imgOut_norm_cuda.download(imgOut_norm_host);
    imwrite("final_images/restored_cuda_256.jpg", imgOut_norm_host);
    /**************** End - Normalization ********************************************/

    printf("\nExecution time: %f seconds\n",elapsedtime.count());

    /**************** Metrics - Image restoration ************************************/
    double psnr1, mse1;
    double psnr2, mse2;
    getPSNR(imgOut_norm_host, imgIn, R, snr, &mse1, &psnr1);
    getPSNR(imgIn, blurIn, R, snr, &mse2, &psnr2);
    printf("\nPerformance Metrics - Image Restoration:");
    printf("\n-----------------------------------------");
    printf("\nBetween Input image and Restored Image");
    printf("\nMean Squared Error: %f | Peak Signal to Noise Ratio: %f",mse1,psnr1);
    printf("\nBetween Input image and Blurred Image");
    printf("\nMean Squared Error: %f | Peak Signal to Noise Ratio: %f\n\n",mse2,psnr2);
}
