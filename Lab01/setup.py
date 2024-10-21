from setuptools import setup, find_packages
from glob import glob

so_files = glob("python/linalg_core*.so")

setup(
    name="linalg",
    version="1.0",
    description="Class provided calculate cosin distanse between two vectors",
    packages=find_packages(),
    package_data={
        "python": ["*.so"],
    },
)