#!/bin/bash

trtexec --onnx=model.onnx \
--saveEngine=model_fp16.plan \
--fp16 \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_fp32.plan \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_int8.plan \
--int8 \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5

trtexec --onnx=model.onnx \
--saveEngine=model_best.plan \
--best \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5
