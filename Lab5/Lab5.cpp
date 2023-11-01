#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const int n = 19;  // Each image in the dataset is 19x19
const int dataSetSize = 400;

int main() {
    string folderPath = filesystem::current_path().string() + "/Lab5/images_faces";

    Mat featuresMatrix(dataSetSize, n * n, CV_8UC1);

    char fname[256];
    for (int k = 1; k <= dataSetSize; ++k) {
        sprintf(fname, "%s/face%05d.bmp", folderPath.c_str(), k);

        Mat img = imread(fname, IMREAD_GRAYSCALE);
        for (int i = 0; i < img.rows; ++i) {
            for (int j = 0; j < img.cols; ++j) {
                featuresMatrix.at<uchar>(k - 1, i * img.cols + j) = img.at<uchar>(i, j);
            }
        }
    }

    vector<double> means(n * n, 0);  // miu
    for (int j = 0; j < featuresMatrix.cols; ++j) {
        int colSum = 0;
        for (int i = 0; i < featuresMatrix.rows; ++i) {
            colSum += (int)featuresMatrix.at<uchar>(i, j);
        }
        means[j] = (double)colSum / dataSetSize;
    }

    vector<double> standardDeviation(n * n, 0);  // sigma
    for (int j = 0; j < featuresMatrix.cols; ++j) {
        double standardDeviationSum = 0;
        for (int i = 0; i < featuresMatrix.rows; ++i) {
            standardDeviationSum += (featuresMatrix.at<uchar>(i, j) - means[j]) * (featuresMatrix.at<uchar>(i, j) - means[j]);
        }
        standardDeviation[j] = sqrt((double)standardDeviationSum / dataSetSize);
    }

    Mat covarianceMatrix = Mat_<double>(n * n, n * n);
    for (int i = 0; i < n * n; ++i) {
        for (int j = 0; j < n * n; ++j) {
            double covarianceSum = 0;
            for (int k = 0; k < dataSetSize; k++) {
                covarianceSum += (featuresMatrix.at<uchar>(k, i) - means[i]) * (featuresMatrix.at<uchar>(k, j) - means[j]);
            }
            covarianceMatrix.at<double>(i, j) = (double)covarianceSum / dataSetSize;
        }
    }

    Mat correlationMatrix = Mat_<double>(n * n, n * n);
    for (int i = 0; i < n * n; ++i) {
        for (int j = 0; j < n * n; ++j) {
            correlationMatrix.at<double>(i, j) = covarianceMatrix.at<double>(i, j) / (standardDeviation[i] * standardDeviation[j]);
        }
    }

    freopen("covariance.txt", "w", stdout);
    for (int i = 0; i < covarianceMatrix.rows; ++i) {
        for (int j = 0; j < covarianceMatrix.cols; ++j) {
            cout << covarianceMatrix.at<double>(i, j) << " ";
        }
        cout << endl;
    }

    freopen("correlation.txt", "w", stdout);
    for (int i = 0; i < correlationMatrix.rows; ++i) {
        for (int j = 0; j < correlationMatrix.cols; ++j) {
            cout << correlationMatrix.at<double>(i, j) << " ";
        }
        cout << endl;
    }

    Mat correlationChart(256, 256, CV_8UC1, Scalar(255));
    const int rowFeature1 = 5, colFeature1 = 4;
    const int rowFeature2 = 5, colFeature2 = 14;

    const int idxFeature1 = rowFeature1 * 19 + colFeature1, idxFeature2 = rowFeature2 * 19 + colFeature2;

    for (int k = 0; k < dataSetSize; ++k) {
        int x = featuresMatrix.at<uchar>(k, idxFeature1);
        int y = featuresMatrix.at<uchar>(k, idxFeature2);
        correlationChart.at<uchar>(x, y) = 0;
    }

    // cout << correlationMatrix.at<double>(idxFeature1, idxFeature2) << endl;

    imshow("Correlation Chart", correlationChart);
    waitKey(0);

    return 0;
}