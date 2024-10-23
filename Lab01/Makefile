CXX=g++

CXXFLAGS=-std=c++11 -O3 -march=native -Wall -I$(SRC_DIR) $(shell python3 -m pybind11 --includes)

PY_LDFLAGS=$(shell python3-config --ldflags) -lopenblas -shared -fPIC

GTEST_FLAGS=-lgtest -lgtest_main -pthread

LDFLAGS=-lopenblas

SRC_DIR   =cosindistance/src
TESTS_DIR =cosindistance/tests
PYTHON_DIR=cosindistance/python

all: CosinDistance test

CosinDistance: $(PYTHON_DIR)/bindings.o $(SRC_DIR)/CosinDistance.o
	$(CXX) $^ -o $(PYTHON_DIR)/cosin_distance`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

$(PYTHON_DIR)/bindings.o: $(PYTHON_DIR)/bindings.cpp $(SRC_DIR)/CosinDistance.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(SRC_DIR)/CosinDistance.o: $(SRC_DIR)/CosinDistance.cpp $(SRC_DIR)/CosinDistance.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

test: $(TESTS_DIR)/testCosinDistance.o $(SRC_DIR)/CosinDistance.o
	$(CXX) $^ -o $(TESTS_DIR)/testCosinDistance $(GTEST_FLAGS) $(LDFLAGS)

$(TESTS_DIR)/testCosinDistance.o: $(TESTS_DIR)/testCosinDistance.cpp $(SRC_DIR)/CosinDistance.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run_tests: test
	./$(TESTS_DIR)/testCosinDistance

clean:
	rm -f $(PYTHON_DIR)/*.o $(SRC_DIR)/*.o $(TESTS_DIR)/*.o $(PYTHON_DIR)/cosin_distance`python3-config --extension-suffix` $(TESTS_DIR)/testCosinDistance