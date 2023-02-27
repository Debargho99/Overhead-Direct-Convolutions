# Efficient Convolution
Implementation of an efficient convolution algorithm between 3D and/or 4D tensors.

## Abstract
The aim of this project is to implement the efficient Direct Convolution algorithm based on the paper [High Performance Zero-Memory Overhead Direct Convolutions][main-paper] by Zhang et al.
The main problem when performing convolutions in deep neural network is that, usually, those higly specialized algorithms trade space for time, incurring in an important memory overhead. The direct convolution could allow us to reduce the memory overhead while keeping performances high.


## Tensor class

The project is entirely base on the `Tensor` class, which allows us to handle 3D and 4D tensor. Those tensors will be used as input images and kernels for the convolution operation.

### Class attributes
```c++
private:
   // Main class members
   T* data;
   uint32_t nElements;
   uint32_t nChannels;
   uint32_t height;
   uint32_t width;
   // Secondary class members
   uint32_t size;
   std::vector<uint32_t> shape;
   bool valid;
```
![](/img/conv.PNG)

### Class constructors

```c++
public:
   // Default constructor
   Tensor();
   // 3D constructor
   Tensor(const uint32_t& nChannels_, const uint32_t& height_, const uint32_t& width_, const tensor::init& init);
   // 4D constructor
   Tensor(const uint32_t& nElements_, const uint32_t& nChannels_, const uint32_t& height_, const uint32_t& width_, const tensor::init& init);
   // Copy constructor
   Tensor(const Tensor<T>& other);
   // Move constructor
   Tensor(Tensor<T>&& other);
```

### Operators-at
The convolution operation is mainly based on the `at()` (`_at()`) operator, that is used in the inner loop of the `convolveThread` method itself and provides high flexibility due to its overloading.

The operator comes in a `public` interface and in a `private` one. The former is more reliable and error-free, while the latter is used for performance issues.
```c++
public:
   // 3D operator at() const
   const T& at(const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx) const;
   // 3D operator at() non-const
   T& at(const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx);

   // 4D operator at() const
   const T& at(const int32_t& E_idx, const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx) const;
   // 4D operator at() non-const
   T& at(const int32_t& E_idx, const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx);
```
```c++
private:
   // 3D operator _at() const
   const T& _at(const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx) const;
   // 3D operator _at() non-const
   T& _at(const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx);

   // 4D operator _at() const
   const T& _at(const int32_t& E_idx, const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx) const;
   // 4D operator _at() non-const
   T& _at(const int32_t& E_idx, const int32_t& C_idx, const int32_t& H_idx, const int32_t& W_idx);

```

### Convolve threads
The convolution operation is parallelized using several threads, each of them performing the convolution on different section of the original `data` pointers (input and kernel tensors). The involved method is `convolveThread` and is implemented in the fashion exposed in the [original paper](main-paper).

### Convolution
Once the `convolveThread` operation is implemented, one can decide the dimension in which to parallelize, whether the number of elments, the number of channels or the height.
![](/img/convolveThread.png)
```c++
public:
   // Convolution operator (parallel) - dimension: output height
   Tensor<T>& convolveParallelHo(const Tensor<T>& kernel, const int32_t stride, const int32_t padding, const uint32_t nThreads) const;
   // Convolution operator (parallel) - dimension: output nChannels
   Tensor<T>& convolveParallelCo(const Tensor<T>& kernel, const int32_t stride, const int32_t padding, const uint32_t nThreads) const;
   // Convolution operator (parallel) - dimension: output nElements
   Tensor<T>& convolveParallelEo(const Tensor<T>& kernel, const int32_t stride, const int32_t padding, const uint32_t nThreads) const;
   
   // Convolution Naive (sequential)
   Tensor<T>& convolveNaive(const Tensor<T>& kernel, const int32_t stride, const int32_t padding) const;

   // Convolution operator that select automatically dimension for parallelization
   Tensor<T>& convolve(const Tensor<T>& kernel, const int32_t stride, const int32_t padding, const uint32_t nThreads) const;
   // Convolution operator that select automatically dimension for parallelization and number of threads
   Tensor<T>& convolve(const Tensor<T>& kernel, const int32_t stride, const int32_t padding) const;

```


## Tests

### Performance of direct convolution against existing high performance FFT-based and SGEMM-based convolution implementations
![](/img/results1.PNG)

### Scaling behavior with increasing number of threads
![](/img/results2.PNG)



## Directory structure

```
.
├── bin
│   ├── benchmark_nopt
│   └── benchmark_opt
├── build
│   ├── benchmark.o
│   ├── Chronometer.o
│   ├── Statistics.o
│   └── Tensor.o
├── doc
│   └── todo.txt
├── include
│   ├── Chronometer.hh
│   ├── Statistics.hh
│   └── Tensor.hh
├── src
│   ├── Chronometer.cpp
│   ├── Statistics.cpp
│   └── Tensor.cpp
├── test
│   ├── benchmark.cpp
│   └── testTensor.cpp
├── Makefile
└── README.md
```

## Documentation and references

[\[1\]][main-paper] Zhang, J., Franchetti, F. &amp; Low, T.M.. (2018). High Performance Zero-Memory Overhead Direct Convolutions. *Proceedings of the 35th International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 80:5776-5785

[\[2\]][concurrency-book] Williams, A. (2019). C++ concurrency in action (Second edition). *Manning Publications Co.*



[main-paper]: http://proceedings.mlr.press/v80/zhang18d/zhang18d.pdf
[concurrency-book]: https://www.manning.com/books/c-plus-plus-concurrency-in-action-second-edition

