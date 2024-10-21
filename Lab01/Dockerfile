FROM ubuntu:20.04

RUN apt update
RUN apt-get install -y g++ python3 libopenblas-dev pip vim git
RUN apt-get install -y libgtest-dev
RUN apt-get install python3.8-venv

RUN pip install setuptools
RUN pip install pybind11
RUN pip install build

RUN git clone https://github.com/pavel-collab/MLOpsLabsSBT.git

WORKDIR /MLOpsLabsSBT/Lab01