#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

pair<pair<int, int>, pair<int, int>> getTwoRandomPoints(vector<pair<int, int>> &points) {
    const int n = points.size();

    pair<int, int> pointOne, pointTwo;
    int idxOne = rand() % n, idxTwo = rand() % n;
    
    while (true) {
        if (idxOne != idxTwo) {
            break;
        }

        idxTwo = rand() % n;
    }

    return {points[idxOne], points[idxTwo]};
}

typedef struct _lineParams {
    int a, b, c;
} lineParams;

lineParams getLineParams(pair<int, int> pointOne, pair<int, int> pointTwo) {
    lineParams params;
    params.a = pointOne.second - pointTwo.second;
    params.b = pointTwo.first - pointOne.first;
    params.c = pointOne.first * pointTwo.second - pointTwo.first * pointOne.second;
    return params;
}

const double thresholdSize = 10;
const double q = 0.3, p = 0.99;
const int N = log(1 - p) / log(1 - q * q);

double getDistanceFromLineToPoint(lineParams lineParams, pair<int, int> point) {
    return (double) abs(lineParams.a * point.first + lineParams.b * point.second + lineParams.c) / ((double) sqrt(lineParams.a + lineParams.a + lineParams.b + lineParams.b));
}

int getConsensusSetSize(vector<pair<int, int>> points, const lineParams &lineParams) {
    int consensusSetSize = 0;
    for (const auto& point : points) {
        auto distance = getDistanceFromLineToPoint(lineParams, point);
        if (distance <= thresholdSize) {
            ++consensusSetSize;
        }
    }

    return consensusSetSize;
}

int main()
{
    string imagePath = filesystem::current_path().string() + "/Lab2/points_RANSAC/points1.bmp";
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);

    vector<pair<int, int>> points;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src.at<uchar>(i, j) == 0) {
                points.push_back({j, i});
            }
        }
    }

    srand(time(NULL));

    int i = 0, bestConsensusSetSize = 0;
    lineParams bestLineParams = {};
    vector<pair<int, int>> bestPoints;
    while (i < N) {
        auto [randomPointOne, randomPointTwo] = getTwoRandomPoints(points);

        lineParams lineParams = getLineParams(randomPointOne, randomPointTwo);
        int consensusSetSize = getConsensusSetSize(points, lineParams);
        
        if (consensusSetSize > bestConsensusSetSize) {
            bestConsensusSetSize = consensusSetSize;
            bestPoints = {randomPointOne, randomPointTwo};
        }

        ++i;
    }

    Point A(bestPoints[0].first, bestPoints[0].second);
    Point B(bestPoints[1].first, bestPoints[1].second);

    line(src, A, B, Scalar(0, 0, 0), 3);
    imshow("points", src);
    waitKey(0);

    return 0;
}