# GGMLSharp Introduction

GGMLSharp is a API for C# to use [ggml](https://github.com/ggerganov/ggml).</br>
ggml is a wonderful C-language machine-learning tool, and now you can use it with C#.</br>
GGMLSharp contains all ggml shared libs and some demos. 

## Feature

- Written in C# only
- Only depends on ggml
- DotNet 462 Support!
- All Demos can use safe code only!

## Demos

### mnist_cpu

  [mnist_cpu](./Demos/MNIST_CPU/) is a basic demo for learning how to use GGMLSharp. It contains two Linears.

### mnist_cnn

  [mnist_cnn](./Demos/MNIST_CNN/) is a demos show how to use convolution. In this demo, there are two conv2d and pool max.

### mnist_train

  [mnist_train](./Demos/MNIST_Train/) is a demo shows how to train a model. The model is same as mnist_cpu.

### simple_backend

  [simple_backend](./Demos/SimpleBackend/) shows how to use GGMLSharp with cuda. In this demo, you shold take ggml.dll for cuda. You can get it with the help of [ggml](https://github.com/ggerganov/ggml) or you can download it from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases).

### magika

[magika](./Demos/Magika/) is a useful tool from google. It can help to get the style of a file in high speed.

### Converter

[Converter](./Demos/Converter/) is a useful tool for converting llm models from bin/ckpt/safetensors to gguf without any python environment. 

### ModelLoader

[ModelLoader](./Demos/ModelLoader/) is a tool for loading safetensors or pickle file directly from binary data. This demo can help to learn how to read a model file without any help of python.

### SAM

[SAM](./Demos/SAM/) (Segment Anything Model) can help us seprate things from an image.

### TestOpt

[TestOpt](./Demos/TestOpt/) is a basic demo for optimizar.

### Yolov3Tiny

[Yolov3Tiny](./Demos/Yolov3Tiny/) is a Demo shows how to implement YOLO object detection with ggml using pretrained model. The weight have been converted to gguf.
