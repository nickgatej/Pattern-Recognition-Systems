CC = g++
PROJECT = output
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
STD = c++17
LIBS = `pkg-config --cflags --libs opencv4`

.PHONY: lab1
lab1:
	@echo "Least Mean Squares Line Fitting"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab1/Lab1.cpp -o $(PROJECT) $(LIBS)

.PHONY: lab2
lab2:
	@echo "RANSAC Line Fitting"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab2/Lab2.cpp -o $(PROJECT) $(LIBS)

.PHONY: lab3
lab3:
	@echo "Hough Transform Line Fitting"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab3/Lab3.cpp -o $(PROJECT) $(LIBS)

.PHONY: lab4
lab4:
	@echo "Distance Transform & Pattern Matching"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab4/Lab4.cpp -o $(PROJECT) $(LIBS)

.PHONY: lab5
lab5:
	@echo "Statistical Data Analysis"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab5/Lab5.cpp -o $(PROJECT) $(LIBS)

.PHONY: lab6
lab6:
	@echo "Principal Component Analysis"
	$(CC) -std=$(STD) $(ROOT_DIR)/Lab6/Lab6.cpp -o $(PROJECT) $(LIBS)	