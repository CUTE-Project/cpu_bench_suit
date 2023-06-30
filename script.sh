#!/bin/bash

echo "stream and oneDNN submodules are Initializing..."
git submodule init
git submodule update
echo "stream and oneDNN submodules are already initialized."

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

# 检查是否需要执行构建和编译操作
if [ "$1" == "make_onnx" ]; then
  cd onnxruntime
  ./build.sh --config Release --use_dnnl --build_wheel --build --update --build_shared_lib --parallel --compile_no_warning_as_error --skip_tests

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
  if [ -s perf.avx512 ]; then
    echo "Execution of cnn_inference_f32_avx512 completed."
    awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.avx512
    awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.avx512
    echo ""
  else
	echo "Execution of cnn_inference_f32_avx512 Failed!."
  fi
  
  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_avx2 > perf.avx2
  if [ -s perf.avx2 ]; then
    echo "Execution of cnn_inference_f32_avx2 completed."
    awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.avx2
    awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.avx2
    echo ""
  else
	echo "Execution of cnn_inference_f32_avx2 Failed!."
  fi

  ONEDNN_VERBOSE=profile_exec ./cnn_inference_f32_sse41 > perf.sse41
  if [ -s perf.sse41 ]; then
    echo "Execution of cnn_inference_f32_sse41 completed."
    awk -F',' '$4 == "inner_product" { sum += $NF; count++ } END { print "The number of inner product layers executed: " count; print "The total duration(ms) is: " sum ; print "Average inner_product time(ms): " sum/count }' ./perf.sse41
    awk -F',' '$4 == "convolution" { sum += $NF; count++ } END { print "The number of convolution layers executed: " count; print "The total duration(ms) is: " sum ; print "Average convolution time(ms): " sum/count }' ./perf.sse41
    echo ""
  else
	echo "Execution of cnn_inference_f32_sse41 Failed!."
  fi

  cp ./perf.* ../../
elif [ "$1" == "report_onednn_more" ]; then
  cd oneDNN/examples
  export DNNLROOT=$(realpath ../build)

  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_avx512.cpp -ldnnl -g -o cnn_inference_f32_avx512
  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_avx2.cpp -ldnnl -g -o cnn_inference_f32_avx2
  g++ -I ${DNNLROOT}/include -I ${DNNLROOT}/../include -Wl,-rpath ${DNNLROOT}/src -L ${DNNLROOT}/src cnn_inference_f32_sse41.cpp -ldnnl -g -o cnn_inference_f32_sse41

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=l1d_Info ./cnn_inference_f32_avx512 > perf_l1d.avx512
  if [ -s perf_l1d.avx512 ]; then
    echo "Execution l1d_Info of cnn_inference_f32_avx512 completed."
    echo ""
  else
	echo "Execution l1d_Info of cnn_inference_f32_avx512 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_W_Info ./cnn_inference_f32_avx512 > perf_llc_w.avx512
  if [ -s perf_llc_w.avx512 ]; then
    echo "Execution LLC_W_Info of cnn_inference_f32_avx512 completed."
    echo ""
  else
	echo "Execution LLC_W_Info of cnn_inference_f32_avx512 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_R_Info ./cnn_inference_f32_avx512 > perf_llc_r.avx512
  if [ -s perf_llc_r.avx512 ]; then
    echo "Execution LLC_R_Info of cnn_inference_f32_avx512 completed."
    echo ""
  else
	echo "Execution LLC_R_Info of cnn_inference_f32_avx512 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=l1d_Info ./cnn_inference_f32_avx2 > perf_l1d.avx2
  if [ -s perf_l1d.avx2 ]; then
    echo "Execution l1d_Info of cnn_inference_f32_avx2 completed."
    echo ""
  else
	echo "Execution l1d_Info of cnn_inference_f32_avx2 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_W_Info ./cnn_inference_f32_avx2 > perf_llc_w.avx2
  if [ -s perf_llc_w.avx2 ]; then
    echo "Execution LLC_W_Info of cnn_inference_f32_avx2 completed."
    echo ""
  else
	echo "Execution LLC_W_Info of cnn_inference_f32_avx2 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_R_Info ./cnn_inference_f32_avx2 > perf_llc_r.avx2
  if [ -s perf_llc_r.avx2 ]; then
    echo "Execution LLC_R_Info of cnn_inference_f32_avx2 completed."
    echo ""
  else
	echo "Execution LLC_R_Info of cnn_inference_f32_avx2 Failed!."
	echo ""
  fi
  
  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=l1d_Info ./cnn_inference_f32_sse41 > perf_l1d.sse41
  if [ -s perf_l1d.sse41 ]; then
    echo "Execution l1d_Info of cnn_inference_f32_sse41 completed."
    echo ""
  else
	echo "Execution l1d_Info of cnn_inference_f32_sse41 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_W_Info ./cnn_inference_f32_sse41 > perf_llc_w.sse41
  if [ -s perf_llc_w.sse41 ]; then
    echo "Execution LLC_W_Info of cnn_inference_f32_sse41 completed."
    echo ""
  else
	echo "Execution LLC_W_Info of cnn_inference_f32_sse41 Failed!."
	echo ""
  fi

  sudo ONEDNN_VERBOSE=profile_exec ONEDNN_VERBOSE_MORE=LLC_R_Info ./cnn_inference_f32_sse41 > perf_llc_r.sse41
  if [ -s perf_llc_r.sse41 ]; then
    echo "Execution LLC_R_Info of cnn_inference_f32_sse41 completed."
    echo ""
  else
	echo "Execution LLC_R_Info of cnn_inference_f32_sse41 Failed!."
	echo ""
  fi

  cp ./perf_l1d.* ../../
  cp ./perf_llc_r.* ../../
  cp ./perf_llc_w.* ../../

elif [ "$1" == "report_stream" ]; then
  cd STREAM
  gcc -Ofast -DSTREAM_ARRAY_SIZE=100000000 -DNTIMES=100 stream.c -o stream.100M
  ./stream.100M > perf.stream
  echo "Execution of stream completed."
  cp ./perf.stream ../
elif [ "$1" == "report_onnx_with_cnn" ]; then
  cd onnxruntime
  export ONNXROOT=$(realpath .)
  mkdir cnn_model
  cd cnn_model
  files=(
    "bvlcalexnet-12.onnx=https://github.com/onnx/models/raw/main/vision/classification/alexnet/model/bvlcalexnet-12.onnx"
    "kitten.jpg=https://s3.amazonaws.com/model-server/inputs/kitten.jpg"
    "resnet34-v1-7.onnx=https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet34-v1-7.onnx"
    "resnet50-v1-12.onnx=https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v1-12.onnx"
    "synset.txt=https://s3.amazonaws.com/onnx-model-zoo/synset.txt"
    "vgg19-7.onnx=https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg19-7.onnx")

	for item in "${files[@]}"; do
		IFS="=" read -r filename url <<< "$item"
		if [ ! -f "$filename" ]; then
			if [ -z "$url" ]; then
				echo "File '$filename' is missing and no download URL is specified."
			else
				echo "Downloading '$filename' from '$url'..."
				wget "$url"
				echo "Download of '$filename' complete."
			fi
		else
			echo "File '$filename' already exists."
		fi
	done

  cd ..
  cp ./cnn_onnx_run.py ./build/Linux/Release/cnn_onnx_run.py
  cd ./build/Linux/Release

  #!/bin/bash

	model_paths=(
	"$ONNXROOT/cnn_model/vgg19-7.onnx"
	"$ONNXROOT/cnn_model/bvlcalexnet-12.onnx"
	"$ONNXROOT/cnn_model/resnet34-v1-7.onnx"
	"$ONNXROOT/cnn_model/resnet50-v1-12.onnx"
	)

	output_files=(
	"onnxperf.vgg19"
	"onnxperf.alexnet"
	"onnxperf.resnet34"
	"onnxperf.resnet50"
	)

	perf_subtest=(
	"l1d_info"
	"llc_w_info"
	"llc_r_info"
	"cycles_info"
	)

	model_paths=(
	"$ONNXROOT/cnn_model/resnet50-v1-12.onnx"
	)

	output_files=(
	"onnxperf.resnet50"
	)

	perf_subtest=(
	"cycles_info"
	)

	for i in "${!model_paths[@]}"; do
	model_path="${model_paths[$i]}"
	for j in "${!perf_subtest[@]}"; do
		output_file="${output_files[$i]}.${perf_subtest[$j]}"
		perf_test="${perf_subtest[$j]}"

		ONEDNN_VERBOSE=1 ONEDNN_VERBOSE_MORE="$perf_test" python cnn_onnx_run.py "$model_path" "$ONNXROOT/cnn_model/synset.txt" "$ONNXROOT/cnn_model/kitten.jpg" > "$output_file"
	done
	done

	cp ./onnxperf.* $ONNXROOT/../

elif [ "$1" == "report_onnx_with_bert" ]; then
  cp -r ./bert_model_requred/* ./onnxruntime/build/Linux/Release/
  export ONNXROOT=$(realpath ./onnxruntime)
  cd onnxruntime/build/Linux/Release
  mkdir output.bert
  perf_subtest=(
	"l1d_info"
	"llc_w_info"
	"llc_r_info"
	"cycles_info"
	)
  python ./run_onnx_squad.py --model ./bert.onnx --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --predict_file ./inputs.json --output ./output.bert

  for j in "${!perf_subtest[@]}"; do
		output_file="onnxperf.bert.${perf_subtest[$j]}"
		perf_test="${perf_subtest[$j]}"
		ONEDNN_VERBOSE=1 ONEDNN_VERBOSE_MORE="$perf_test" python ./run_onnx_squad.py --model ./bert.onnx --vocab_file ./uncased_L-12_H-768_A-12/vocab.txt --predict_file ./inputs.json --output ./output.bert > ./output.bert/"$output_file"
  done
  cp ./output.bert/* $ONNXROOT/../
fi

