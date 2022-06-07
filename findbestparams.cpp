#include <math.h>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>

// C++ Auto-Tune OpenCV Implementation of Image restoration 
// Helps to determine the best values of R and snr for a given image when performing PSF function based image restoration 
// Compile: g++ findbestparams.cpp -o bestparams -I/usr/local/include/opencv4 -lopencv_core  -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc  $(pkg-config opencv4 --libs)
// Run: ./bestparams

using namespace cv;
using namespace std;
void help();
void calcPSF(Mat& outputImg, Size filterSize, int R);
void fftshift(const Mat& inputImg, Mat& outputImg);
void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);
void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);
const String keys =
"{help h usage ? |             | print this message   }"
"{image          |original.JPG | input image name     }"
"{R              |53           | radius               }"
"{SNR            |5200         | signal to noise ratio}"
;

struct filter_metrics {
    int R;
    int snr;
    double mse;
    double psnr;
}filter;

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
        double temp_mse =sse /(double)(I1.channels() * I1.total());
        double temp_psnr = 10.0*log10((255*255)/temp_mse);
        if(temp_psnr > psnr){
            filter.psnr = temp_psnr;
            filter.mse = temp_mse;
            filter.R = R;
            filter.snr = snr;
        }
    }
    return 0;
}

int main(int argc, char *argv[])
{
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

    GaussianBlur(imgIn, blurIn, Size(5,5), 0);

    filter.psnr = 0.0f;
    filter.mse = 0.0f;
    
    for(int R = 2; R < 7; R ++){
        for(int snr = 50; snr < 5000; snr+=500){
            Mat imgOut;
            // To process even image only
            Rect roi = Rect(0, 0, blurIn.cols & -2, blurIn.rows & -2);
            //Hw calculation (start)
            Mat Hw, h;
            calcPSF(h, roi.size(), R);
            calcWnrFilter(h, Hw, 1.0 / double(snr));
            filter2DFreq(blurIn(roi), imgOut, Hw);
            imgOut.convertTo(imgOut, CV_8U);
            normalize(imgOut, imgOut, 0, 255, NORM_MINMAX);
            getPSNR(imgIn,imgOut, R, snr, filter.mse, filter.psnr);
        }
    }
    cout << "AutoTuned - Best values:" << endl;
    cout << "Radius (R): " << filter.R << endl;
    cout << "Signal to Noise Ratio (snr): " << filter.snr << endl;

    
    cout << "\nPeak Signal to Noise Ratio (PSNR): " << filter.psnr << endl;
    cout << "Mean Squared Error (MSE): " << filter.mse << endl;
    return 0;
}
void help()
{
    cout << "2018-07-12" << endl;
    cout << "DeBlur_v8" << endl;
    cout << "You will learn how to recover an out-of-focus image by Wiener filter" << endl;
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