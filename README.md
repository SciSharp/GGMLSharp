# GGMLSharp Introduction

GGMLSharp is a API for C# to use [ggml](https://github.com/ggerganov/ggml).</br>
ggml is a wonderful C-language machine-learning tool, and now you can use it with C#.</br>
GGMLSharp contains all ggml shared libs and some demos. 

## Feature

- Written in C# only
- Only depends on ggml

## Demos

### mnist_cpu

  [mnist_cpu](./examples/mnist_cpu/) is a basic demo for learning how to use GGMLSharp. It contains two Linears.

### mnist_cnn

  [mnist_cnn](./examples/mnist_cnn/) is a demos show how to use convolution. In this demo, there are two conv2d and pool max.

### mnist_train

  [mnist_train](./examples/mnist_train/) is a demo shows how to train a model. The model is same as mnist_cpu.

### simple_backend

  [simple_backend](./examples/simple_backend/) shows how to use GGMLSharp with cuda. In this demo, you shold take ggml.dll for cuda. You can get it with the help of [ggml](https://github.com/ggerganov/ggml) or you can download it from [llama.cpp](https://github.com/ggerganov/llama.cpp/releases).

### magika

[magika](./examples/magika/) is a useful tool from google. It can help to get the style of a file in high speed.

### Converter

[Converter](./examples/Converter/) is a useful tool for converting llm models from bin/ckpt/safetensors to gguf without any python environment. And in this demo you can also get how to read bin and safetensors file from binary data with stream.
