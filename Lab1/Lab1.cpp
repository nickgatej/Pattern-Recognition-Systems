#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const int height = 1000;
const int width = 1000;

int main()
{
    string filePath = filesystem::current_path().string() + "/Lab1/points/points1.txt";
    freopen(filePath.c_str(), "r", stdin);

    int n;
    cin >> n;

    vector<pair<double, double>> points;
    points.reserve(n);

    for (int i = 0; i < n; ++i) {
        double x, y;
        cin >> x >> y;
        points.push_back(make_pair(x, y));
    }

    Mat img(height, width, CV_8UC3, Scalar(255, 255, 255));
    
    for (const auto& point : points) {
        img.at<Vec3b>(point.second, point.first) = Vec3b(0, 0, 0);
    }

    double sumXY = 0, sumX = 0, sumY = 0, sumX2 = 0, sumY2X2 = 0;
    for (const auto& point : points) {
        sumXY += point.first * point.second;
        sumX += point.first;
        sumY += point.second;
        sumX2 += point.first * point.first;
        sumY2X2 += (point.second * point.second - point.first * point.first);
    }

    double theta_1 = (double) (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    double theta_0 = (double) (sumY - theta_1 * sumX) / n;

    Point A(0, theta_0);
    Point B(255, theta_1 * 255 + theta_0);
    line(img, A, B, Scalar(0, 0, 255), 3);

    double beta = (-0.5) * atan2(2 * sumXY  - (2.0 / n) * (sumX * sumY), sumY2X2 + (1.0 / n) * sumX * sumX - (1.0 / n) * sumY * sumY);
    double rho = (1.0 / n) * (cos(beta) * sumX + sin(beta) * sumY);

    if (abs(beta) > CV_PI / 4) {
        Point A(0, rho / (sin(beta)));
        Point B(width, (rho - width * cos(beta)) / (sin(beta)));
        line(img, A, B, Scalar(0, 255, 0), 3); 
    } else {
        Point A(rho / (cos(beta)), 0);
        Point B((rho - height * sin(beta)) / (cos(beta)), height);
        line(img, A, B, Scalar(0, 255, 0), 3); 
    }

    imshow("points", img);
    waitKey(0);

    return 0;
}