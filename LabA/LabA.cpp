#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const int maxIterations = 100000;
const double learningRate = 0.0001, errorLimit = 0.0001;

int main() {
    string imagePath = filesystem::current_path().string() + "/LabA/images_Perceptron/test01.bmp";

    Mat_<Vec3b> img = imread(imagePath, IMREAD_COLOR);

    Mat_<int> X(0, 3);
    Mat_<int> y(0, 1);
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            int row[] = {1, j, i};
            Mat_<int> rowMatrix(1, 3, row);
            if (img(i, j) == Vec3b(0, 0, 255)) {
                X.push_back(rowMatrix);
                y.push_back(-1);
            } else if (img(i, j) == Vec3b(255, 0, 0)) {
                X.push_back(rowMatrix);
                y.push_back(1);
            }
        }
    }

    vector<double> w(3, 0.1);
    const int n = X.rows;

    for (int it = 0; it < maxIterations; ++it) {
        int errorCount = 0;
        for (int i = 0; i < n; ++i) {
            double z = 0;
            for (int j = 0; j < 3; ++j) {
                z += w[j] * X(i, j);
            }

            if (z * y(i, 0) <= 0) {
                for (int j = 0; j < 3; j++) {
                    w[j] = w[j] + learningRate * y(i, 0) * X(i, j);
                }
                ++errorCount;
            }
        }

        double errorPercentage = (double)errorCount / n;
        if (errorPercentage <= errorLimit) {
            break;
        }
    }

    cout << "Weights: " << endl;
    for (int i = 0; i < 3; i++) {
        cout << "w[" << i << "] = " << w[i] << endl;
    }

    double w0 = w[0], w1 = w[1], w2 = w[2];

    Point p1;
    p1.x = 0;
    p1.y = -(w0 + w1 * p1.x) / w2;

    Point p2;
    p2.x = img.cols;
    p2.y = -(w0 + w1 * p2.x) / w2;

    line(img, p1, p2, Scalar(128, 0, 128), 1);
    imshow("Perceptron", img);
    waitKey(0);

    return 0;
}