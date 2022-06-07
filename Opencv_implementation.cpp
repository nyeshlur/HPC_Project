#include <math.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

// C++ OpenCV Implementation of Image restoration
// Compile: g++ Opencv_implementation.cpp -o OpenCV -I/usr/local/include/opencv4 -lopencv_core  -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc  $(pkg-config opencv4 --libs)
// Run: ./OpenCV

using namespace cv;
using namespace std;

void calcPSF(Mat& outputImg, Size filterSize, int R);
void fftshift(const Mat& inputImg, Mat& outputImg);
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

int main(int argc, char *argv[])
{
    int R = 2; // Radius of PSF function
    int snr = 105;
    double psnr = 0.0f;
    double mse  = 0.0f;

    Mat imgIn;
    imgIn = imread("final_images/bear_256.png", IMREAD_GRAYSCALE);

    // Making sure the images are of square size and even sized
    if(imgIn.size().width > imgIn.size().height)
        if(imgIn.size().height % 2 == 0)
            imgIn = imgIn(Range(0,imgIn.size().height), Range(0,imgIn.size().height));
        else
            imgIn = imgIn(Range(0,imgIn.size().height-1), Range(0,imgIn.size().height-1));
    else if(imgIn.size().height > imgIn.size().width)
        if(imgIn.size().width % 2 == 0)
            imgIn = imgIn(Range(0,imgIn.size().width), Range(0,imgIn.size().width));
        else
            imgIn = imgIn(Range(0,imgIn.size().width-1), Range(0,imgIn.size().width-1));
    Mat blurIn = imgIn.clone();

    GaussianBlur(imgIn, blurIn, Size(5,5), 0);
    imwrite("images/blur.jpg", blurIn);

    std::chrono::time_point<std::chrono::system_clock> startInc, endInc, startExc, endExc;
    startInc = std::chrono::system_clock::now();

    Mat imgOut;
    // To process even image only
    Rect roi = Rect(0, 0, blurIn.cols & -2, blurIn.rows & -2);
    Mat Hw, h;
    calcPSF(h, roi.size(), R);
    calcWnrFilter(h, Hw, 1.0 / double(snr));
    filter2DFreq(blurIn(roi), imgOut, Hw);
    imgOut.convertTo(imgOut, CV_8U);
    normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);

    imwrite("images/restored_orig.jpg", imgOut);
    endInc = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsedtime_inc = endInc - startInc;

    // Metrics - Image restoration
    printf("\nExclusive time: %f seconds\n",elapsedtime_inc.count());

    printf("\nPerformance Metrics - Image Restoration:");
    printf("\n-----------------------------------------");
    printf("\nBetween Input image and Restored Image");
    psnr = getPSNR(imgIn,imgOut, R, snr, mse, psnr);
    printf("\nPeak Signal to Noise Ratio: %f\n\n",psnr);
    return 0;
}

/*********** STAGE 1 *******************************************************/
/********** Creating the Point Spread Function (PSF) ***********************/
void calcPSF(Mat& outputImg, Size filterSize, int R)
{
    Mat h(filterSize, CV_32F, Scalar(0));
    Point point(filterSize.width / 2, filterSize.height / 2);
    circle(h, point, R, 255, -1, 8);
    Scalar summa = sum(h);
    outputImg = h / summa[0];
}

/*********** STAGE 2 *******************************************/
/********** Creating the Wiener Filter  ***********************/
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
    Mat h_PSF_shifted;
    fftshift(input_h_PSF, h_PSF_shifted);
    Mat planes[2] = { Mat_<float>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    //cout << complexI << endl;
    dft(complexI, complexI);
    split(complexI, planes);
    Mat denom;
    pow(abs(planes[0]), 2, denom);
    denom += nsr;
    divide(planes[0], denom, output_G);
}

// FFT Shift of Point Spread Function
void fftshift(const Mat& inputImg, Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    Mat q0(outputImg, Rect(0, 0, cx, cy));
    Mat q1(outputImg, Rect(cx, 0, cx, cy));
    Mat q2(outputImg, Rect(0, cy, cx, cy));
    Mat q3(outputImg, Rect(cx, cy, cx, cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

/*********** STAGE 3 *******************************************/
/********** Creating the Restored Image  ***********************/
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
    Mat planes[2] = { Mat_<float>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI, DFT_SCALE);
    Mat planesH[2] = { Mat_<float>(H.clone()), Mat::zeros(H.size(), CV_32F) };
    Mat complexH;
    merge(planesH, 2, complexH);
    Mat complexIH;
    mulSpectrums(complexI, complexH, complexIH, 0);
    idft(complexIH, complexIH);
    split(complexIH, planes);
    outputImg = planes[0];
}
