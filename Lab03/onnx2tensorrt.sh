#!/bin/bash

trtexec --onnx=model.onnx \
--saveEngine=model_fp16.plan \
--minShapes=input:1 \
--optShapes=input:8 \
--maxShapes=input:8 \
--inputIOFormats=fp32:chw \
--fp16 \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_fp32.plan \
--minShapes=input:1 \
--optShapes=input:8 \
--maxShapes=input:8 \
--inputIOFormats=fp32:chw \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_int8.plan \
--minShapes=input:1 \
--optShapes=input:8 \
--maxShapes=input:8 \
--inputIOFormats=fp32:chw \
--int8 \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_best.plan \
--minShapes=input:1 \
--optShapes=input:8 \
--maxShapes=input:8 \
--inputIOFormats=fp32:chw \
--best \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5