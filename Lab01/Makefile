CXX=g++

CXXFLAGS=-std=c++11 -O3 -march=native -Wall -I$(SRC_DIR) $(shell python3 -m pybind11 --includes)

PY_LDFLAGS=$(shell python3-config --ldflags) -lopenblas -shared -fPIC

# sudo apt-get install libopenblas-dev need for this flag
LDFLAGS=-lopenblas

SRC_DIR=./src
PYTHON_DIR=./python

all: linalg


# linalg: $(PYTHON_DIR)/bindings.o $(SRC_DIR)/LinAlg.o
# 	$(CXX) $^ -o $(PYTHON_DIR)/linalg_core`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

linalg: $(PYTHON_DIR)/bindings.o $(SRC_DIR)/LinAlg.o
	$(CXX) $^ -o $(PYTHON_DIR)/linalg_core.so $(PY_LDFLAGS) $(CXXFLAGS)

$(PYTHON_DIR)/bindings.o: $(PYTHON_DIR)/bindings.cpp $(SRC_DIR)/LinAlg.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(SRC_DIR)/LinAlg.o: $(SRC_DIR)/LinAlg.cpp $(SRC_DIR)/LinAlg.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

# clean:
# 	rm -f $(PYTHON_DIR)/*.o $(SRC_DIR)/*.o $(TESTS_DIR)/*.o $(PYTHON_DIR)/linalg_core`python3-config --extension-suffix` $(TESTS_DIR)/test_linalg

clean:
	rm -f $(PYTHON_DIR)/*.o $(SRC_DIR)/*.o $(TESTS_DIR)/*.o $(PYTHON_DIR)/linalg_core.so $(TESTS_DIR)/test_linalg