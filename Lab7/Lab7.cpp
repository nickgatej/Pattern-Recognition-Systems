#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

const int k = 5;
default_random_engine engine;

vector<pair<int, int>> getPoints(const Mat& src) {
    vector<pair<int, int>> points;
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src.at<uchar>(i, j) == 0) {
                points.push_back({j, i});
            }
        }
    }
    return points;
}

set<pair<int, int>> getKRandomPoints(const vector<pair<int, int>>& points) {
    int pointsSelected = 0;
    const int n = points.size();

    set<pair<int, int>> kRandomPoints;
    uniform_int_distribution<int> distribution(0, n - 1);
    while (pointsSelected < k) {
        int randomIdx = distribution(engine);
        if (kRandomPoints.count(points[randomIdx])) {
            continue;
        }
        kRandomPoints.emplace(points[randomIdx]);
        ++pointsSelected;
    }

    return kRandomPoints;
}

int main() {
    string imagePath = filesystem::current_path().string() + "/Lab7/images_Kmeans/points5.bmp";
    Mat src = imread(imagePath, IMREAD_GRAYSCALE);

    vector<Vec3b> colors(k);
    uniform_int_distribution<int> distribution(0, 255);
    for (int i = 0; i < k; ++i) {
        colors[i] = {(uchar)distribution(engine), (uchar)distribution(engine), (uchar)distribution(engine)};
    }

    vector<pair<int, int>> points = getPoints(src);
    set<pair<int, int>> kRandomPoints = getKRandomPoints(points);

    vector<pair<int, int>> kPoints;
    for (const auto& randomPoint : kRandomPoints) {
        kPoints.push_back(randomPoint);
    }

    vector<int> pointsClusterIdx(points.size());  // [pointIdx -> clusterIdx]
    while (true) {
        for (int i = 0; i < (int)points.size(); ++i) {
            int bestDistance = INT_MAX;
            for (int j = 0; j < (int)kPoints.size(); ++j) {
                double distance = sqrt((points[i].first - kPoints[j].first) * (points[i].first - kPoints[j].first) + (points[i].second - kPoints[j].second) * (points[i].second - kPoints[j].second));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    pointsClusterIdx[i] = j;
                }
            }
        }

        bool changeOccured = false;
        for (int clusterIdx = 0; clusterIdx < k; ++clusterIdx) {
            int sumX = 0, sumY = 0, counter = 0;
            for (int i = 0; i < (int)pointsClusterIdx.size(); ++i) {
                if (clusterIdx == pointsClusterIdx[i]) {
                    sumX += points[i].first;
                    sumY += points[i].second;
                    ++counter;
                }
            }

            pair<int, int> potentialNewCenter = {sumX / counter, sumY / counter};
            if (potentialNewCenter.first != kPoints[clusterIdx].first || potentialNewCenter.second != kPoints[clusterIdx].second) {
                changeOccured = true;
                kPoints[clusterIdx].first = potentialNewCenter.first;
                kPoints[clusterIdx].second = potentialNewCenter.second;
            }
        }

        if (!changeOccured) {
            break;
        }
    }

    Mat clusters(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < (int)points.size(); ++i) {
        int colorIdx = pointsClusterIdx[i];
        clusters.at<Vec3b>(points[i].first, points[i].second) = colors[colorIdx];
    }

    imshow("Clusters", clusters);

    Mat tesselation(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < tesselation.rows; ++i) {
        for (int j = 0; j < tesselation.cols; ++j) {
            int bestDistance = INT_MAX;
            for (int clusterIdx = 0; clusterIdx < (int)kPoints.size(); ++clusterIdx) {
                double distance = sqrt((i - kPoints[clusterIdx].first) * (i - kPoints[clusterIdx].first) + (j - kPoints[clusterIdx].second) * (j - kPoints[clusterIdx].second));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    tesselation.at<Vec3b>(i, j) = colors[clusterIdx];
                }
            }
        }
    }

    imshow("Voronoi Tesselation", tesselation);
    waitKey(0);

    return 0;
}