#!/bin/bash

# 检查当前目录是否存在stream和oneDNN文件夹
if [ ! -d "STREAM" ] || [ ! -d "oneDNN" ]; then
  # 初始化submodule
  git submodule init
fi

if git submodule status STREAM >/dev/null 2>&1 && git submodule status oneDNN >/dev/null 2>&1; then
  echo "stream and oneDNN submodules are already initialized."
else
  echo "stream and oneDNN submodules are not initialized. Initializing..."
  git submodule init
  git submodule update
fi

# 检查是否需要执行构建和编译操作
if [ "$1" == "make" ]; then
  cd oneDNN
  export DNNLROOT=$(pwd)
  mkdir build
  cd build

  # 编译和构建
  cmake .. -DCMAKE_BUILD_TYPE=Debug -DDNNL_CPU_RUNTIME=SEQ
  make -j16

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
  awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.avx512
  awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.avx512
  echo ""
  
  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_avx2 > perf.avx2
  echo "Execution of cnn_inference_f32_avx2 completed."
  awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.avx2
  awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.avx2
  echo ""

  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_sse41 > perf.sse41
  echo "Execution of cnn_inference_f32_sse41 completed."
  awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.sse41
  awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.sse41
  echo ""

  cp ./perf.* ../../
elif [ "$1" == "report_stream" ]; then
  cd STREAM
  gcc -Ofast -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=100 stream.c -o stream.100M
  ./stream.100M > perf.stream
  echo "Execution of stream completed."
  cp ./perf.stream ../
fi

