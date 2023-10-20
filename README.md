# cpu bench suit

The submodule of this repository is a collection of 'state of the art' machine learning software stacks.

This repository counts the performance of the processor during the execution of each primitive.We register PMUs before and after the execution of neural network primitives and let PMUs 'pin' to get the precise microarchitecture information at the time of execution.

#### Usage

`./script.sh make` Compiles the oneDNN library in a single thread.

`./script.sh make_onnx` Compile onnxruntime and use oneDNN library as execution backend (use openmp as multithreaded library).

`./script.sh report_onednn_more` outputs performance information for each layer of alexnet(fp32) execution on the oneDNN library, focusing on the differences between the different vector instruction sets chosen for expansion.

`./script.sh report_onnx_with_cnn` Executes different cnn models on the onnxruntime and outputs layer-by-layer performance information for each cnn model.

`./script.sh report_cpu_infrence_latency` Execute resnet50 on onnxruntime and count top99%,top90% 1batch inference latency

the script in oneDNN，We prepared two scripts for xeon 6338 and xeon 8475B (supported instruction set, data width, avoiding hyperthreading congestion). It is used to count the best execution of oneDNN when executing each convolutional layer of vgg19 and resnet50.

`cd oneDNN`

`./conv_test_6338.sh`

`./conv_test_8475B.sh`
