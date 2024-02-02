#include <bits/stdc++.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#define THRESHOLD 127
#define MAX_PATH 1024

using namespace std;
using namespace cv;

char fname[MAX_PATH];
const int numOfClasses = 10, d = 28, trainingFiles = 60000; // each image has size d * d
const vector<string> classes = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};

vector<double> computeAprioriProbabilities(vector<int> imgCount) {
    vector<double> apriori(numOfClasses);
    for (int i = 0; i < numOfClasses; ++i) {
        apriori[i] = (double)imgCount[i] / trainingFiles;
    }

    return apriori;
}

Mat_<double> computeLikelihood(Mat_<double> X, Mat_<int> y, vector<int> imgCount) {
    Mat_<double> likelihood(numOfClasses, d * d);
    likelihood.setTo(1);
    for (int i = 0; i < X.rows; ++i) {
        int classIdx = y(i, 0);
        for (int j = 0; j < X.cols; ++j) {
            if (X(i, j) == 255) {
                likelihood(classIdx, j)++;
            }
        }
    }

    // Laplace Smoothing
    for (int i = 0; i < numOfClasses; ++i) {
        double denominator = imgCount[i] + numOfClasses;
        for (int j = 0; j < d * d; ++j) {
            likelihood(i, j) = likelihood(i, j) / denominator;
        }
    }

    return likelihood;
}

vector<double> computePosteriorProbabilities(Mat_<double> X, vector<double> apriori, Mat_<double> likelihood, int rowIdx) {
    vector<double> posteriorProbabilities(numOfClasses, 0);
    for (int i = 0; i < numOfClasses; ++i) {
        posteriorProbabilities[i] = log(apriori[i]);
        for (int j = 0; j < d * d; ++j) {
            if (X(rowIdx, j) == 255) {
                posteriorProbabilities[i] += log(likelihood(i, j));
            } else {
                posteriorProbabilities[i] += log(1 - likelihood(i, j));
            }
        }
    }
    return posteriorProbabilities;
}

pair<Mat_<double>, Mat_<int>> train(string folderPath, vector<int> &imgCount) {
    Mat_<double> X(trainingFiles, d * d);  // feature matrix
    Mat_<int> y(trainingFiles, 1);         // label vector

    int rowIdx = 0;
    for (int classIdx = 0; classIdx < numOfClasses; ++classIdx) {
        int pictureIdx = 0;
        while (true) {
            sprintf(fname, "%s/train/%s/%06d.png", folderPath.c_str(), classes[classIdx].c_str(), pictureIdx++);

            Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
            if (img.cols == 0) {
                break;
            }

            imgCount[classIdx]++;
            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    X(rowIdx, i * img.cols + j) = img(i, j) > THRESHOLD ? 255 : 0;
                }
            }

            y.at<int>(rowIdx) = classIdx;
            ++rowIdx;
        }
    }

    return pair<Mat_<double>, Mat_<int>>{X, y};
}

void classifyBayes(vector<double> apriori, Mat_<double> likelihood, string folderPath) {
    Mat_<double> X(trainingFiles, d * d);  // feature matrix
    Mat_<int> y(trainingFiles, 1);         // label vector

    int rowIdx = 0, correctAnswersCount = 0, wrongAnswersCount = 0;
    for (int classIdx = 0; classIdx < numOfClasses; ++classIdx) {
        int pictureIdx = 0;

        while (true) {
            sprintf(fname, "%s/test/%s/%06d.png", folderPath.c_str(), classes[classIdx].c_str(), pictureIdx++);

            Mat_<uchar> img = imread(fname, IMREAD_GRAYSCALE);
            if (img.cols == 0) {
                break;
            }

            for (int i = 0; i < img.rows; ++i) {
                for (int j = 0; j < img.cols; ++j) {
                    X(rowIdx, i * img.cols + j) = img(i, j) > THRESHOLD ? 255 : 0;
                }
            }

            vector<double> posteriorProbabilities = computePosteriorProbabilities(X, apriori, likelihood, rowIdx);

            int predictedClassIdx = 0;
            double predictedClassProbability = posteriorProbabilities[0];
            for (int i = 1; i < numOfClasses; ++i) {
                if (posteriorProbabilities[i] > predictedClassProbability) {
                    predictedClassIdx = i;
                    predictedClassProbability = posteriorProbabilities[i];
                }
            }

            y.at<int>(rowIdx) = classIdx;
            ++rowIdx;

            // classIdx is the actualClass
            classIdx == predictedClassIdx ? ++correctAnswersCount : ++wrongAnswersCount;
        }
    }

    cout << "Correct Answers: " << correctAnswersCount << endl;
    cout << "Wrong Answers: " << wrongAnswersCount << endl;
    cout << "Accuracy: " << (double)correctAnswersCount / (correctAnswersCount + wrongAnswersCount) << endl;
}

int main() {
    string folderPath = filesystem::current_path().string() + "/Lab9/images_Bayes";

    vector<int> imgCount(numOfClasses, 0); // imgCount[i] = num of pictures in class i
    auto [X, y] = train(folderPath, imgCount);

    vector<double> apriori = computeAprioriProbabilities(imgCount);
    Mat_<double> likelihood = computeLikelihood(X, y, imgCount);
    classifyBayes(apriori, likelihood, folderPath);

    return 0;
}