include .build.env.default

.PHONY: setup
setup:
	@echo "Setting up the project"
	mkdir -p $(BUILD_FOLDER) && meson setup $(BUILD_FOLDER)

.PHONY: build
build:
	@echo "Building the project"
	meson compile -C $(BUILD_FOLDER)

.PHONY: lab1
lab1:
	@echo "Least Mean Squares Line Fitting"
	./$(BUILD_FOLDER)/Lab1

.PHONY: lab2
lab2:
	@echo "RANSAC Line Fitting"
	./$(BUILD_FOLDER)/Lab2

.PHONY: lab3
lab3:
	@echo "Hough Transform Line Fitting"
	./$(BUILD_FOLDER)/Lab3

.PHONY: lab4
lab4:
	@echo "Distance Transform & Pattern Matching"
	./$(BUILD_FOLDER)/Lab4

.PHONY: lab5
lab5:
	@echo "Statistical Data Analysis"
	./$(BUILD_FOLDER)/Lab5

.PHONY: lab6
lab6:
	@echo "Principal Component Analysis"
	./$(BUILD_FOLDER)/Lab6

.PHONY: lab7
lab7:
	@echo "K-Means Clustering"
	./$(BUILD_FOLDER)/Lab7

.PHONY: lab8
lab8:
	@echo "K Nearest Neighbors"
	./$(BUILD_FOLDER)/Lab8

.PHONY: lab9
lab9:
	@echo "Naive Bayes Classifier"
	./$(BUILD_FOLDER)/Lab9

.PHONY: clean
clean:
	@echo "Removing build folder"
	rm -rf $(BUILD_FOLDER)
