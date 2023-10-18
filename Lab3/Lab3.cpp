#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

#define all(x) (x).begin(), (x).end()

const vector<int> dirRows = {-1, -1, -1, 0, 0, 1, 1, 1};
const vector<int> dirCols = {-1, 0, 1, -1, 1, -1, 0, 1};

struct peak {
    int theta, rho, hval;
    bool operator < (const peak& o) const {
        return hval > o.hval;
    }
};

int main()
{
    string imagePath = filesystem::current_path().string() + "/Lab3/images_Hough/edge_simple.bmp";
    Mat srcGrayscale = imread(imagePath, IMREAD_GRAYSCALE);
    Mat srcColored = imread(imagePath, IMREAD_COLOR);
    
    const int n = srcGrayscale.rows, m = srcGrayscale.cols;
    const int D = sqrt(n * n + m * m); // rho max

    vector<pair<int, int>> points;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (srcGrayscale.at<uchar>(i, j) == 255) {
                points.push_back({j, i});
            }
        }
    }

    Mat Hough(D + 1, 360, CV_32SC1);
    int HoughMax = INT_MIN;

    Hough.setTo(0);
    for (const auto& [x, y] : points) {
        for (int theta = 0; theta < 360; ++theta) {
            double angle = theta * CV_PI / 180;
            double rho = x * cos(angle) + y * sin(angle);
            if (0 <= rho && rho <= D) {
                Hough.at<int>(rho, theta)++;
                HoughMax = max(HoughMax, Hough.at<int>(rho, theta));
            }
        }
    }

    Mat houghImg;
    Hough.convertTo(houghImg, CV_8UC1, 255.f / HoughMax);

    vector<peak> peaks;
    for (int i = 0; i < Hough.rows; ++i) {
        for (int j = 0; j < Hough.cols; ++j) {
            int windowMax = INT_MIN;
            for (int k = 0; k < 8; ++k) {
                int row = i + dirRows[k];
                int col = j + dirCols[k];
                bool isInBoundaries = row >= 0 && col >= 0 && row < Hough.rows && col < Hough.cols;
                if (isInBoundaries) {
                    windowMax = max(windowMax, Hough.at<int>(row, col));
                }
            }

            if (Hough.at<int>(i, j) > windowMax) {
                peaks.push_back(peak{j, i, Hough.at<int>(i, j)});
            }
        }
    }

    sort(all(peaks));

    int numLines = 10;
    for (const auto& peak : peaks) {
        if (numLines == 0) {
            break;
        }

        const double rho = peak.rho, theta = peak.theta * CV_PI / 180;
        const int width = srcColored.cols, height = srcColored.rows;

        if (abs(theta) > CV_PI / 4) {
            Point A(0, rho / (sin(theta)));
            Point B(width, (rho - width * cos(theta)) / (sin(theta)));
            line(srcColored, A, B, Scalar(0, 255, 0), 1); 
        } else {
            Point A(rho / (cos(theta)), 0);
            Point B((rho - height * sin(theta)) / (cos(theta)), height);
            line(srcColored, A, B, Scalar(0, 255, 0), 1); 
        }
        --numLines;
    }

    imshow("Hough Space", houghImg);
    imshow("Hough Transform", srcColored);
    waitKey(0);


    return 0;
}