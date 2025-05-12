## 0. requirements
- ubuntu > 20.04
- cmake > 3.16 (개발환경 : 3.26)
- c++17
- onnxruntime > 1.19.0 (개발환경 : 1.21.0)
```
$ cd cpp_colution
$ wget https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/onnxruntime-linux-x64-1.21.0.tgz
$ tar -xvf onnxruntime-linux-x64-1.21.0.tgz
```
- rustc, cargo
```
$ sudo apt install rustc cargo
```
- tokenizers_cpp
```
$ git clone --recurse-submodules https://github.com/mlc-ai/tokenizers-cpp.git
```
- opencv (개발환경 4.4.0)

## 1. build
```
$ cd cpp_solution
$ mkdir build && cd build
$ cmake ..
$ make -j8
```

## 2. llm_vlm_cpp_solution

### 2-1. llm_cpp
- execute
```
$ cd cpp_solution/build
$ ./llm_solution/llm_cpp_solution [model.onnx] [tokenizer.json]
$ ./llm_solution/llm_cpp_solution ../llm/model/q4f16.onnx ../llm/tokenizer/tokenizer.json
```

### 2-2. convert model
- access 허가 : https://huggingface.co/google/gemma-3-1b-it
- huggingface login
- 가상환경
```
$ sudo apt  install python3.10 python3.10-venv python3.10-dev
$ python3.10 -m venv venv_cm
$ source venv_cm/bin/activate
$ pip install torch transformers onnx
$ pip install optimum[exporters]
```
- execute
```
$ cd model_convert
$ python model_converter.py
```


### 2-3. vlm_cpp
- prepare tokenizer (merge_tokens.py 내부, 경로수정)
```
$ cd cpp_solution
$ python3 merge_tokens.py
```
- execute
```
$  cd cpp_solution/build
$ ./vlm_solution/vlm_cpp_solution [vision_encoder.onnx] [token_embedding.onnx] [decoder.onnx] [tokenizer.json] [image_path]
$ ./vlm_solution/vlm_cpp_solution ../vlm/model/vision_encoder.onnx ../vlm/model/token_embedding.onnx ../vlm/model/decoder.onnx ../vlm/merged_tokenizer/tokenizer.json ../assets/test_image.png
```

## directory tree
```
.
├── README.md
├── cpp_solution
│   ├── CMakeLists.txt
│   ├── assets
│   ├── build
│   ├── llm
│   ├── llm_solution
│   ├── merge_tokens.py
│   ├── onnxruntime-linux-x64-1.21.0
│   ├── onnxruntime-linux-x64-1.21.0.tgz
│   ├── tokenizers-cpp
│   ├── utils
│   ├── vlm
│   └── vlm_solution
├── model_convert
│   └── model_converter.py
└── venv_cm
```