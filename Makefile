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
