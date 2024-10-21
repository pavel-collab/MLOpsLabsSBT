#include "CosinDistance.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cosin_distance, m) {
  m.doc() = R"doc(
    Python bindings for CosinDistance library
  )doc";

  py::class_<CosinDistance>(m, "CosinDistance")
      .def_static("GetVectorNorm", &CosinDistance::GetVectorNorm, R"doc(
          Compute vector norm using pure C++.

          Parameters:
            a : list of float
                The vector.

          Returns:
            float
                Norm of the vector.
      )doc")
      .def_static("GetCosDistance", &CosinDistance::GetCosDistance, R"doc(
          Compute cosin distance between two vectors using pure C++.

          Parameters:
            a : list of float
                The first vector.
            b : list of float
                The second vector.

          Returns:
            float
                Cosin distance between two vectors.
      )doc");
}