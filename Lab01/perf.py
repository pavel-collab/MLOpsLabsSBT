import numpy as np 
from scipy.spatial import distance
import time

from cosindistance import CosinDistance


def test_timings(func: callable, *args):
    _ = func(*args)
    start_time = time.time()
    _ = func(*args)
    end_time = time.time()
    return round(end_time - start_time, 5)


if __name__ == "__main__":
    for length in [5, 25, 125, 625, 3125, 15625, 78125]:
        vec1 = np.random.rand(length)
        vec2 = np.random.rand(length)
        print("Cosin distance (Pure C++),     size={0}: {1} seconds".format(length, test_timings(CosinDistance.get_cos_distance, vec1, vec2)))
        print("Cosin distance (Python scipy), size={0}: {1} seconds\n".format(length, test_timings(distance.cosine, vec1, vec2)))