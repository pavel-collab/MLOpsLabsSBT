#include "CosinDistance.hpp"

#include <numeric>
// #include <functional>
#include <cassert>
#include <cmath>

double CosinDistance::get_vector_norm(const std::vector<double> &vec)
{
    double res = 0;
    res = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0);
    return res;
}

double CosinDistance::get_cos_distance(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    //! to be honest we have to use smth like exceptions here,
    //! but for this task temporary let it be
    assert(vec1.size() == vec2.size());

    double vec1_norm = std::sqrt(CosinDistance::get_vector_norm(vec1));
    double vec2_norm = std::sqrt(CosinDistance::get_vector_norm(vec2));

    assert(vec1_norm != 0);
    assert(vec2_norm != 0);

    float scalar_multiplication = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);

    return scalar_multiplication / (vec1_norm * vec2_norm);
}