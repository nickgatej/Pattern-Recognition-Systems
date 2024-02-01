#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define MAX_PATH 1024

using namespace std;
using namespace cv;

char fname[MAX_PATH];
const int numOfClasses = 6, K = 5, binsCount = 8, trainingFiles = 672;
const vector<string> classes = {"beach", "city", "desert", "forest", "landscape", "snow"};

vector<vector<int>> getHistogram(Mat_<Vec3b> img, int binsCount) {
    vector<int> histogramRed(binsCount, 0);
    vector<int> histogramGreen(binsCount, 0);
    vector<int> histogramBlue(binsCount, 0);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.cols; ++j) {
            Vec3b pixel = img(i, j);

            int redBin = pixel[2] * binsCount / 256;
            int greenBin = pixel[1] * binsCount / 256;
            int blueBin = pixel[0] * binsCount / 256;

            histogramRed[redBin]++;
            histogramGreen[greenBin]++;
            histogramBlue[blueBin]++;
        }
    }

    return vector<vector<int>>{{histogramRed}, {histogramGreen}, {histogramBlue}};
}

pair<Mat_<double>, Mat_<int>> train(string folderPath) {
    Mat_<double> X(trainingFiles, 3 * binsCount);  // feature matrix
    Mat_<int> y(trainingFiles, 1);                 // label vector

    int rowIdx = 0;
    for (int classIdx = 0; classIdx < numOfClasses; ++classIdx) {
        int pictureIdx = 0;
        while (true) {
            char fname[MAX_PATH];
            sprintf(fname, "%s/train/%s/%06d.jpeg", folderPath.c_str(), classes[classIdx].c_str(), pictureIdx++);

            Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
            if (img.cols == 0) {
                break;
            }

            vector<vector<int>> histogram = getHistogram(img, binsCount);
            for (int binIdx = 0; binIdx < binsCount; ++binIdx) {
                X.at<double>(rowIdx, binIdx) = histogram[0][binIdx];
                X.at<double>(rowIdx, binIdx + binsCount) = histogram[1][binIdx];
                X.at<double>(rowIdx, binIdx + 2 * binsCount) = histogram[2][binIdx];
            }

            y.at<int>(rowIdx) = classIdx;
            ++rowIdx;
        }
    }

    return pair<Mat_<double>, Mat_<int>>{X, y};
}

vector<pair<double, int>> computeEuclideanDistances(Mat_<double> fv, Mat_<double> X, Mat_<int> y) {
    vector<pair<double, int>> distances;
    for (int i = 0; i < X.rows; ++i) {
        double distance = 0;
        for (int j = 0; j < X.cols; ++j) {
            distance += (fv(0, j) - X(i, j)) * (fv(0, j) - X(i, j));
        }
        distance = sqrt(distance);
        distances.push_back({distance, y(i, 0)});  // pair: dist, idx of the class
    }
    return distances;
}

void classifyImages(Mat_<double> X, Mat_<int> y, string folderPath) {
    Mat_<int> ConfusionMatrix(numOfClasses, numOfClasses, 0);

    int correctAnswersCount = 0, wrongAnswersCount = 0;
    for (int classIdx = 0; classIdx < numOfClasses; ++classIdx) {
        int pictureIdx = 0;

        while (true) {
            char fname[MAX_PATH];
            sprintf(fname, "%s/test/%s/%06d.jpeg", folderPath.c_str(), classes[classIdx].c_str(), pictureIdx++);

            Mat_<Vec3b> img = imread(fname, IMREAD_COLOR);
            if (img.cols == 0) {
                break;
            }

            vector<vector<int>> histogram = getHistogram(img, binsCount);
            Mat_<double> featureVector = Mat_<double>(1, 3 * binsCount);
            for (int binIdx = 0; binIdx < binsCount; ++binIdx) {
                featureVector.at<double>(0, binIdx) = histogram[0][binIdx];
                featureVector.at<double>(0, binIdx + binsCount) = histogram[1][binIdx];
                featureVector.at<double>(0, binIdx + 2 * binsCount) = histogram[2][binIdx];
            }

            vector<pair<double, int>> distances = computeEuclideanDistances(featureVector, X, y);
            sort(distances.begin(), distances.end());

            // take K Nearest Neigbors and find the Neighbour with the most number of "votes"
            vector<int> votes(numOfClasses);
            for (int i = 0; i < K; ++i) {
                int classIdx = distances[i].second;
                votes[classIdx]++;
            }

            int predictedClassIdx = 0, predictedClassVotes = votes[0];
            for (int i = 1; i < numOfClasses; ++i) {
                if (votes[i] > predictedClassVotes) {
                    predictedClassIdx = i;
                    predictedClassVotes = votes[i];
                }
            }

            // classIdx is the actualClass
            classIdx == predictedClassIdx ? ++correctAnswersCount : ++wrongAnswersCount;
            ConfusionMatrix(classIdx, predictedClassIdx)++;
        }
    }

    cout << "Correct Answers: " << correctAnswersCount << endl;
    cout << "Wrong Answers: " << wrongAnswersCount << endl;
    cout << "Accuracy: " << (double)correctAnswersCount / (correctAnswersCount + wrongAnswersCount) << endl;
    cout << "Confusion matrix: \n";
    cout << ConfusionMatrix << endl;
}

int main() {
    string folderPath = filesystem::current_path().string() + "/Lab8/images_KNN";

    auto [X, y] = train(folderPath);
    classifyImages(X, y, folderPath);

    return 0;
}