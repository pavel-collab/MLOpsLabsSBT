#include "CosinDistance.hpp"

#include <cassert>
#include <cmath>

double CosinDistance::get_vector_norm(const std::vector<double> &vec)
{
    double res = 0;
    for (auto el : vec) 
        res += el*el;
    return res;
}

double CosinDistance::get_cos_distance(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    //TODO: what if one of the vectors is 0
    assert(vec1.size() == vec2.size());

    float scalar_multiplication = 0;
    for (std::size_t i = 0; i < vec1.size(); i++)
        scalar_multiplication += vec1[i] * vec2[i];
    
    double vec1_norm = std::sqrt(CosinDistance::get_vector_norm(vec1));
    double vec2_norm = std::sqrt(CosinDistance::get_vector_norm(vec2));

    return scalar_multiplication / (vec1_norm * vec2_norm);
}