#pragma once

#include <vector>

class LinAlg {
public:
    static double GetCosDistance(const std::vector<double> &vec1, 
                                 const std::vector<double> &vec2);

    static double GetVectorNorm(const std::vector<double> &vec);
};