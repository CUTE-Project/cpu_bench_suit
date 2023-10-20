# cpu bench suit

The current submodule of the repository is a collection of 'state of the art' machine learning software stacks.

This repository counts the performance of the processor during the execution of each primitive by calling linux system calls, embedding the resident state pmu and registering specific events before and after the execution of the neural network primitive.

#### Usage

`. /script.sh make` Compiles the oneDNN library in a single thread.

`. /script.sh make_onnx` Compile onnxruntime and use oneDNN library as execution backend (use openmp as multithreaded library).

`. /script.sh report_onednn_more` outputs performance information for each layer of alexnet(fp32) execution on the oneDNN library, focusing on the differences between the different vector instruction sets chosen for expansion.

`. /script.sh report_onnx_with_cnn` Executes different cnn models on the onnxruntime and outputs layer-by-layer performance information for each cnn model.

`. /script.sh report_cpu_infrence_latency` Execute resnet50 on onnxruntime and count top99%,top90% 1batch inference latency
