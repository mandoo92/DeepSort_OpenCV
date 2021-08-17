# DeepSort CPP Only OpenCV_DNN

A refactoring version of [DeepSort_TensorRT](https://github.com/GesilaA/deepsort_tensorrt)

Remove TensorRT from the repository and inference using OpenCV DNN

# Requirements

OpenCV>=4.2.0 With CUDA(Tested by 4.5.1)

Eigen(Tested by 3.3.9)

# Install

It was tested based on Visual Studio 2019, but I think it will work well enough with Cmake.

~~~
mkdir build
cd build
cmake ..
make
~~~

# PreTrained Model
reference [DeepSort_PyTorch](https://github.com/ZQPei/deep_sort_pytorch)

I converted the model(ckpt.t7) downloaded from Github above to Onnx

[Google Drive Link(deepsort.onnx)](https://drive.google.com/file/d/1eTKZBaSilFZV2z6SjkJzQMrCg7wnaXco/view?usp=sharing)

# ToDO
- [ ] Detect Model Add




