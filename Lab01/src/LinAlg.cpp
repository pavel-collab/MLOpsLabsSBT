#include "LinAlg.hpp"

#include <cassert>
#include <cmath>

//TODO: maybe it can be more optimal with using std::asynch https://ru.stackoverflow.com/questions/1263021/
double LinAlg::GetVectorNorm(const std::vector<double> &vec)
{
    double res = 0;
    for (auto el : vec) 
        res += el*el;
    return res;
}

double LinAlg::GetCosDistance(const std::vector<double> &vec1, const std::vector<double> &vec2)
{
    assert(vec1.size() == vec2.size());

    float scalar_multiplication = 0;
    for (std::size_t i = 0; i < vec1.size(); i++)
        scalar_multiplication += vec1[i] * vec2[i];
    
    double vec1_norm = std::sqrt(LinAlg::GetVectorNorm(vec1));
    double vec2_norm = std::sqrt(LinAlg::GetVectorNorm(vec2));

    return scalar_multiplication / (vec1_norm * vec2_norm);
}