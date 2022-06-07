#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

// C++ Sequential Implementation of Image restoration (Manual - without Opencv's built-in functions)
// Compile: g++ Sequential_implementation.cpp -o sequential -I/usr/local/include/opencv4 -lopencv_core  -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc  $(pkg-config opencv4 --libs)
// Run: ./sequential

using namespace cv;
using namespace std;
void calcPSF(Mat& outputImg, Size filterSize, int R);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);

/********** Calculate correctness of algorithm using MSE and PSNR ****************************/
/* PSNR - Peak Signal to noise Ratio
   MSE - Mean Squared Error */
double getPSNR(const Mat& I1, const Mat& I2, int R, int snr, double mse, double psnr)
{
    Mat s1;
    absdiff(I1, I2, s1);
    s1.convertTo(s1, CV_32F);
    s1 = s1.mul(s1);

    Scalar s = sum(s1);
    double sse = s.val[0] + s.val[1] + s.val[2];

    if( sse <= 1e-10)
        return 0;
    else
    {
        double mse =sse /(double)(I1.channels() * I1.total());
        double psnr = 10.0*log10((255*255)/mse);
        printf("\nMean Squared Error: %f",mse);
        return psnr;
    }
}

int main()
{
    int R = 2; // Radius of PSF function
    int snr = 105;
    double psnr = 0.0f;
    double mse  = 0.0f;

    Mat imgIn;
    imgIn = imread("final_images/bear_256.png", IMREAD_GRAYSCALE);
    Mat blurIn = imgIn.clone();

    GaussianBlur(imgIn, blurIn, Size(5,5), 0);
    imwrite("final_images/blur_seq.jpg", blurIn);

    Mat imgOut, Out;

    // To process even image only
    Rect roi = Rect(0, 0, blurIn.cols & -2, blurIn.rows & -2);
    Mat Hw, h(roi.height, roi.width, CV_32FC1);
    std::chrono::time_point<std::chrono::system_clock> startInc, endInc, startExc, endExc;
    startInc = std::chrono::system_clock::now();
    calcPSF(h, roi.size(), R);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    filter2DFreq(blurIn(roi), imgOut, Hw);
    imgOut.convertTo(Out, CV_8U);

    //Find min and max of image
    int max = Out.at<uchar>(0, 0);
    int min = Out.at<uchar>(0, 0);

    for(int i=0;i<Out.rows;i++){
        for(int j=0;j<Out.cols;j++){
            if(Out.at<uchar>(i, j) > max)
                    max = Out.at<uchar>(i, j);
            else if(Out.at<uchar>(i, j) < min)
                    min = Out.at<uchar>(i, j);
        }
    }

    //Normalizing the final restored image
    for(int i=0;i<Out.rows;i++){
        for(int j=0;j<Out.cols;j++){
            int tmp = round((Out.at<uchar>(i, j) - min) * (255.0/(max-min)));
            Out.at<uchar>(i, j) = tmp;
        }
    }
    imwrite("final_images/restored_seq.jpg", Out);

    endInc = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedtime_inc = endInc - startInc;

    // Metrics - Image restoration
    printf("\nExecution time: %f seconds\n",elapsedtime_inc.count());
    printf("\nPerformance Metrics - Image Restoration:");
    printf("\n-----------------------------------------");
    printf("\nBetween Input image and Restored Image");
    psnr = getPSNR(imgIn,Out, R, snr, mse, psnr);
    printf("\nPeak Signal to Noise Ratio: %f\n\n",psnr);

    return 0;
}

/*********** STAGE 1 *******************************************************/
/********** Creating the Point Spread Function (PSF) ***********************/
void calcPSF(Mat& outputImg, Size filterSize, int radius)
{
    int size = filterSize.height;
    int midx = size/2;
    int midy = size/2;

    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            outputImg.at<float>(i, j) = 0.0;
        }
    }
    double summa =0.0;
    for(int y=-radius; y<=radius; y++){
        for(int x=-radius; x<=radius; x++){
            if(x*x+y*y <= radius*radius){
                outputImg.at<float>(midx+x, midy+y) = 255;
                summa += 255;
            }
        }
    }

    if(summa!=0){
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                if(outputImg.at<float>(i,j)!=0.0f){
                    outputImg.at<float>(i,j) = outputImg.at<float>(i,j)/summa;
                }
            }
        }
    }
    return;
}

/*********** STAGE 2 *******************************************/
/********** Creating the Wiener Filter  ***********************/
void calcWnrFilter(const Mat& input_PSF, Mat& output_G, double nsr)
{
    // FFT Shift of Point Spread Function
    Mat PSF_shifted = input_PSF.clone();
    int cx = input_PSF.cols / 2;
    int cy = input_PSF.rows / 2;
    Mat Q0(cx,cy,CV_32FC1),Q1(cx,cy,CV_32FC1),Q2(cx,cy,CV_32FC1),Q3(cx,cy,CV_32FC1);
    int i,j;
    for(i=0;i<cx;i++){
        for(j=0;j<cy;j++){
            Q0.at<float>(i,j) = input_PSF.at<float>(i, j);
            Q1.at<float>(i,j) = input_PSF.at<float>(i, j+cy);
            Q2.at<float>(i,j) = input_PSF.at<float>(i+cx, j);
            Q3.at<float>(i,j) = input_PSF.at<float>(i+cx, j+cy);
        }
    }
    for(i=0;i<cx;i++){
        for(j=0;j<cy;j++){
            PSF_shifted.at<float>(i,j) = Q3.at<float>(i,j);
            PSF_shifted.at<float>(i,j+cy) = Q2.at<float>(i,j);
            PSF_shifted.at<float>(i+cx,j) = Q1.at<float>(i,j);
            PSF_shifted.at<float>(i+cx,j+cy) = Q0.at<float>(i,j);
        }
    }

    Mat planes[2] = { Mat_<float>(PSF_shifted.clone()), Mat::zeros(PSF_shifted.size(), CV_32F) };
    Mat complexI(input_PSF.rows,input_PSF.cols,CV_32FC2);

    // Merge Filter with zero values plane - Complex data type
    for(int i=0;i<input_PSF.rows;i++){
        for(int j=0;j<input_PSF.cols;j++){
            float tmp = PSF_shifted.at<float>(i,j);
            complexI.at<std::complex<float> >(i,j) =  std::complex<float>(tmp,0);
        }
    }

    // Discrete Fourier Transform using existing OpenCV's functionality
    dft(complexI, complexI);

    // Split Filter containing two planes to a single plane
    for(int i=0; i < input_PSF.rows; i++){
        for(int j=0; j < input_PSF.cols; j++){
            planes[0].at<float>(i,j) = complexI.at<float>(i,j+j);
        }
    }
    // Adding and Division as part of restoration process
    Mat denom_div(input_PSF.rows,input_PSF.cols,CV_32FC1);
    for(i=0;i<input_PSF.rows;i++){
        for(j=0;j<input_PSF.cols;j++){
            denom_div.at<float>(i,j) = nsr + (abs(planes[0].at<float>(i,j))*abs(planes[0].at<float>(i,j)));
            denom_div.at<float>(i,j) = planes[0].at<float>(i,j)/denom_div.at<float>(i,j);
        }
    }
    output_G = denom_div.clone();
}

/*********** STAGE 3 *******************************************/
/********** Creating the Restored Image  ***********************/
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI(inputImg.rows,inputImg.cols,CV_32FC2);

    // Merge Image with zero values plane - Complex data type
    for(int i=0;i<inputImg.rows;i++){
        for(int j=0;j<inputImg.rows;j++){
            int tmp = inputImg.at<uchar>(i,j);
            complexI.at<std::complex<float> >(i,j) =  std::complex<float>(tmp,0);
        }
    }

    // Discrete Fourier Transform using existing OpenCV's functionality
    dft(complexI, complexI, DFT_SCALE);

    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH(inputImg.rows,inputImg.cols,CV_32FC2);

    // Merge Filter with zero values plane - Complex data type
    for(int i=0;i<inputImg.rows;i++){
        for(int j=0;j<inputImg.rows;j++){
            float tmp = H.at<float>(i,j);
            complexH.at<std::complex<float> >(i,j) =  std::complex<float>(tmp,0);
        }
    }
    Mat complexIH(inputImg.rows,inputImg.cols,CV_32FC2);

    // Dot product of image and wiener filter - Equivalent of 'Mulspectrums' in OpenCV
    for(int i=0;i<inputImg.rows;i++){
        for(int j=0;j<inputImg.rows;j++){
            float I_real = complexI.at<std::complex<float> >(i,j).real();
            float I_img = complexI.at<std::complex<float> >(i,j).imag();
            float H_real = complexH.at<std::complex<float> >(i,j).real();
            float mul_real = I_real * H_real;
            float mul_imag = I_img * H_real;
            complexIH.at<std::complex<float> >(i,j) =  std::complex<float>(mul_real,mul_imag);
        }
    }

    // Inverse Discrete Fourier Transform using existing OpenCV's functionality
    idft(complexIH, complexIH);

    // Split Image/Filter containing two planes to a single plane
    for(int i=0; i < inputImg.rows; i++){
        for(int j=0; j < inputImg.cols; j++){
            planes[0].at<float>(i,j) = complexIH.at<float>(i,j+j);
        }
    }
    outputImg = planes[0];
}
