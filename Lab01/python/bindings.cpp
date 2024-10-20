#include "LinAlg.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(linalg_core, m) {
  m.doc() = R"doc(
    Python bindings for LinAlg library
  )doc";

  py::class_<LinAlg>(m, "LinAlg")
      .def_static("GetVectorNorm", &LinAlg::GetVectorNorm, R"doc(
          Compute vector norm using pure C++.

          Parameters:
            a : list of float
                The vector.

          Returns:
            float
                Norm of the vector.
      )doc")
      .def_static("GetCosDistance", &LinAlg::GetCosDistance, R"doc(
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