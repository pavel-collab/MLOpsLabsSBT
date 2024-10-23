#include "CosinDistance.hpp"
#include <gtest/gtest.h>

TEST(CosinDistanceTests, VectorNorm) {
  std::vector<double> a = {1.0, 2.0, 3.0};

  EXPECT_TRUE(std::abs(CosinDistance::get_vector_norm(a) - 14.0) < 1e-2);
}

TEST(CosinDistanceTests, CosinDistance) {
  std::vector<double> a = {1.0, 2.0, 3.0};
  std::vector<double> b = {4.0, 5.0, 6.0};

  EXPECT_TRUE(std::abs(CosinDistance::get_cos_distance(a, b) - 0.97463184619707621) < 1e-6);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}