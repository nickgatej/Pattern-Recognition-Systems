#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const int k = 2;

vector<double> getMeanValues(Mat matrix, const int& n, const int& m) {
    vector<double> mu;
    for (int col = 0; col < m; ++col) {
        double featureSum = 0;
        for (int row = 0; row < n; ++row) {
            featureSum += matrix.at<double>(row, col);
        }
        mu.push_back(featureSum / n);
    }
    return mu;
}

int main() {
    string filePath = filesystem::current_path().string() + "/Lab6/data_PCA/pca2d.txt";
    freopen(filePath.c_str(), "r", stdin);

    int patterns, features;
    cin >> patterns >> features;

    Mat featuresMatrix(patterns, features, CV_64FC1);  // double

    for (int row = 0; row < patterns; ++row) {
        for (int col = 0; col < features; ++col) {
            double feature;
            cin >> feature;
            featuresMatrix.at<double>(row, col) = feature;
        }
    }

    vector<double> mu = getMeanValues(featuresMatrix, patterns, features);

    Mat X(patterns, features, CV_64FC1);
    for (int i = 0; i < patterns; ++i) {
        for (int j = 0; j < features; ++j) {
            X.at<double>(i, j) = featuresMatrix.at<double>(i, j) - mu[j];
        }
    }

    Mat covarianceMatrix = (X.t() * X) / (patterns - 1);

    Mat_<double> Lambda, Q;  // Q contains eigenvectors and Lambda eigenvalues
    eigen(covarianceMatrix, Lambda, Q);
    Q = Q.t();

    // for (int i = 0; i < 7; ++i) {
    //     cout << Lambda.at<double>(i, 0) << " ";
    // }

    Mat Q1k(features, k, CV_64FC1);
    for (int i = 0; i < features; ++i) {
        for (int j = 0; j < k; ++j) {
            Q1k.at<double>(i, j) = Q.at<double>(i, j);
        }
    }

    Mat XCoeff = X * Q1k;
    double minColOne = 1e7, minColTwo = 1e7, maxColOne = -1e7, maxColTwo = -1e7;
    for (int i = 0; i < XCoeff.rows; ++i) {
        for (int j = 0; j < XCoeff.cols; ++j) {
            if (j == 0) {
                minColOne = min(minColOne, XCoeff.at<double>(i, j));
                maxColOne = max(maxColOne, XCoeff.at<double>(i, j));
            }

            if (j == 1) {
                minColTwo = min(minColTwo, XCoeff.at<double>(i, j));
                maxColTwo = max(maxColTwo, XCoeff.at<double>(i, j));
            }
        }
    }

    Mat_<uchar> plot(maxColOne - minColOne + 1, maxColTwo - minColTwo + 1);
    plot.setTo(255);
    for (int i = 0; i < patterns; ++i) {
        plot(XCoeff.at<double>(i, 0) - minColTwo, XCoeff.at<double>(i, 1) - minColOne) = 0;
    }

    Mat XTilda = XCoeff * Q1k.t();
    double MAD = 0;
    for (int i = 0; i < patterns; ++i) {
        for (int j = 0; j < features; ++j) {
            MAD += abs(XTilda.at<double>(i, j) - X.at<double>(i, j));
        }
    }
    MAD /= (features * patterns);

    cout << MAD << endl;

    imshow("Plot", plot);
    waitKey(0);

    return 0;
}