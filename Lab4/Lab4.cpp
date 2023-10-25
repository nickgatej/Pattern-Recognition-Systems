#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const vector<int> maskTLRows = {-1, -1, -1, 0, 0};  // | | |
const vector<int> maskTLCols = {-1, 0, 1, -1, 0};   // | *

const vector<int> maskBRRows = {0, 0, 1, 1, 1};   //   * |
const vector<int> maskBRCols = {0, 1, -1, 0, 1};  // | | |

const int diagonalPixelWeight = 3;
const int nonDiagonalPixelWeight = 2;

bool isInBoundaries(const int &row, const int &col, const int &n, const int &m) {
    return row >= 0 && col >= 0 && row < n && col < m;
}

Mat computeDistanceTransform(const Mat &src) {
    const int n = src.rows, m = src.cols;

    Mat distanceTransform = src.clone();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < 5; ++k) {
                int newRow = i + maskTLRows[k];
                int newCol = j + maskTLCols[k];

                int cellWeight = (k == 0 || k == 2) ? diagonalPixelWeight : nonDiagonalPixelWeight;
                if (newRow == i && newCol == j) {
                    cellWeight = 0;
                }

                if (isInBoundaries(newRow, newCol, n, m)) {
                    distanceTransform.at<uchar>(i, j) = min(
                        (int)distanceTransform.at<uchar>(i, j),
                        (int)distanceTransform.at<uchar>(newRow, newCol) + cellWeight);
                }
            }
        }
    }

    for (int i = n - 1; i >= 0; --i) {
        for (int j = m - 1; j >= 0; --j) {
            for (int k = 0; k < 5; ++k) {
                int newRow = i + maskBRRows[k];
                int newCol = j + maskBRCols[k];

                int cellWeight = (k == 2 || k == 4) ? diagonalPixelWeight : nonDiagonalPixelWeight;
                if (newRow == i && newCol == j) {
                    cellWeight = 0;
                }

                if (isInBoundaries(newRow, newCol, n, m)) {
                    distanceTransform.at<uchar>(i, j) = min(
                        (int)distanceTransform.at<uchar>(i, j),
                        (int)distanceTransform.at<uchar>(newRow, newCol) + cellWeight);
                }
            }
        }
    }

    return distanceTransform;
}

double computeMatchingScore(const Mat &contour, const Mat &unknownObject, const Mat &distanceTransform) {
    double matchingScore = 0;
    int objectPixels = 0;
    for (int i = 0; i < unknownObject.rows; ++i) {
        for (int j = 0; j < unknownObject.cols; ++j) {
            if (unknownObject.at<uchar>(i, j) == 0) {
                ++objectPixels;
                matchingScore += distanceTransform.at<uchar>(i, j);
            }
        }
    }

    matchingScore /= objectPixels;
    return matchingScore;
}

int main() {
    string templatePath = filesystem::current_path().string() + "/Lab4/images_DT_PM/PatternMatching/template.bmp";
    string unknownObjectImagePath = filesystem::current_path().string() + "/Lab4/images_DT_PM/PatternMatching/unknown_object1.bmp";

    Mat src = imread(templatePath, IMREAD_GRAYSCALE);
    Mat unknownObject = imread(unknownObjectImagePath, IMREAD_GRAYSCALE);
    Mat distanceTransform = computeDistanceTransform(src);

    imshow("Distance Transform", distanceTransform);
    cout << computeMatchingScore(src, unknownObject, distanceTransform) << endl;

    waitKey(0);

    return 0;
}