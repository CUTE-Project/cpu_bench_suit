#!/bin/bash

# 检查当前目录是否存在stream和oneDNN文件夹
if [ ! -d "stream" ] || [ ! -d "oneDNN" ]; then
  # 初始化submodule
  git submodule init
fi

# 检查是否需要执行构建和编译操作
if [ "$1" == "make" ]; then
  cd oneDNN
  export DNNLROOT=$(pwd)
  mkdir build
  cd build

  # 编译和构建
  cmake .. -DCMAKE_BUILD_TYPE=Debug -DDNNL_CPU_RUNTIME=SEQ
  make -j8

  # 返回上级目录
  cd ../

  exit 0
fi

# 检查是否需要执行report操作
if [ -z "$1" ]; then
  read -p "It will make oneDNN and stream (take some time and occupy CPU) [Y/n]: " choice
  if [[ $choice == "Y" || $choice == "y" ]]; then
    ./script.sh make
    if [ $? -eq 0 ]; then
      echo "Making oneDNN completed successfully."
    else
      echo "Failed to make oneDNN. Exiting..."
      exit 1
    fi

    read -p "Do you want to proceed with reporting? [Y/n]: " report_choice
    if [[ $report_choice == "Y" || $report_choice == "y" ]]; then
      ./script.sh report_onednn
      ./script.sh report_stream
    else
      exit 0
    fi
  elif [[ $choice == "N" || $choice == "n" ]]; then
    exit 0
  else
    echo "Invalid choice. Please try again."
  fi
elif [ "$1" == "report_onednn" ]; then
  cd oneDNN/examples
  export DNNLROOT=$(realpath ../build)

  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_avx512.cpp -ldnnl -g -o cnn_inference_f32_avx512
  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_avx2.cpp -ldnnl -g -o cnn_inference_f32_avx2
  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_sse41.cpp -ldnnl -g -o cnn_inference_f32_sse41

  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_avx512 > perf.avx512
  echo "Execution of cnn_inference_f32_avx512 completed."

  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_avx2 > perf.avx2
  echo "Execution of cnn_inference_f32_avx2 completed."
  
  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_sse41 > perf.sse41
  echo "Execution of cnn_inference_f32_sse41 completed."

  cp ./perf.* ../../
elif [ "$1" == "report_stream" ]; then
  cd stream
  gcc -Ofast -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=100 stream.c -o stream.100M
  ./stream.100M > perf.stream
  echo "Execution of stream completed."
  cp ./perf.stream ../
fi

