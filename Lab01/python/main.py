import linalg_core

from scipy.spatial import distance

vec1 = [1, 2, 3]
vec2 = [4, 5, 6]

print(linalg_core.LinAlg.GetVectorNorm(vec1))
print(linalg_core.LinAlg.GetVectorNorm(vec2))

print(linalg_core.LinAlg.GetCosDistance(vec1, vec2))
print(1 - distance.cosine(vec1, vec2)) # note that in the original scipy library this distance is the 1 - cos()