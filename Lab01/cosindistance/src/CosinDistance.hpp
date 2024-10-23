#pragma once

#include <vector>

class CosinDistance {
public:
    static double get_cos_distance(const std::vector<double> &vec1, 
                                 const std::vector<double> &vec2);

    static double get_vector_norm(const std::vector<double> &vec);
};