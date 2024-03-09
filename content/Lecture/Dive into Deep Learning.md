# Chapter 01
# Chapter 02
# Chapter 03
# Chapter 04
# Chapter 05
# CH 08 Modern Convolutional Neural Networks

Modern CNN
- AlexNet
- VGG network
- NiN(Network in Network)
- GoogLeNet
- ResNet
- DenseNet

## AlexNet(Deep Convolutional Neural Networks)

### LeNet

- they did not immediately dominate the field
- good results on early small dataset
    - larger, more realistic dataset had yet to be established
- they were not yet sufficiently powerful to make deep multichannel, multilayer
- conception(not yet)
    - parameter initialization
    - gradient descent
    - regularization
- End-toEnd X → classical pippelines
    - intersting dataset
    - preprocess the dataset
    - set(standard set of feature extractors) dataset
    - resulting representations → favorite classifier

### AlexNet

- 8-layer CNN(CONV_5 + MaxPooling_3) + 2FC + 1output(FC)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/1ad22e9c-6e90-4cb5-9105-98c08cfc0f67/Untitled.png)

1. much deeper than the comparatively small LeNet-5
2. ReLu instead of sigmoid as its activation function
    1. vanishing gradient 문제 해결
3. Convolutional Block
    1. 11×11 strides=4 Conv
        1. larger convolution window is needed to capture the object
    2. 3×3 strides=2 MaxPool
    3. 5×5 padding=2 Conv
    4. 3×3 stride=2 MaxPool
    5. 3×3 paddding=1 Conv 레이어 3개
    6. 3×3 stride=2 MaxPool
4. adds max pooling layers
5. 10-times more convolution channels than LeNet
6. two huge fully connected layer with 4096 outputs
    1. two GPU
    2. nowadays we rarely need to break up models
7. dropout ↔ LeNet uses weight decay
    1. 과적합 방지
8. data augmentation

## VGG

### Networks Using Blocks

- block → repeating patterns of layers
    - conv layer + maxpooling layer
- visual geometry group(VGG)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/ba41d444-3394-4646-86dd-7444d4ab90c2/Untitled.png)

- basic building block of CNN
    - a convolutional layer with padding to maintain the resolution
    - nonlinearity → ReLu
    - Pooling layer → max-pooling
        - to reduce the resolution
- → spatial resolution decrease quite rapidly
- VGG → use multiple convolutions in between downsampling via max-pooling in the form of a block
- the successive application of two 3_3 convolutions touches the same pixels as a single 5_5 convolution does

## Network in Network

- LeNet, AlexNet VGG → a common design pattern
    - extract features exploiting _spatial_ structure via a sequence of convolutions and pooling layers and post-process the representations via fully connected layers
- two major challenges
    - the fully connected layers at the end of the architecture
        - consume tremendous numbers of parameters
    - impossible to add fully connected layers earlier in the network to increase the degree of nonlinearity
- NiN Network in Network
    - use 1*1 convolutions to add local nonlinearities across the channel activations
    - use global average pooling to integrate across all locations in the last representation layer

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/2ea7e0fc-04a7-47fe-ac2e-ec87af84ef16/Untitled.png)

구조를 보면 NiN은 AlexNet에 착안하여 만들어졌기에 conv window의 size가 AlexNet과 그 쌍이 같음을 알 수 있다. 하지만 둘 사이의 큰 차이가 존재하는데 NiN은 fc layer을 사용하지 않았고 대신에 NiN block에서 output channel의 갯수가 label class의 갯수와 같게 설정하였고 global average pooling layer가 존재한다. 이러한 NiN 구조의 장점은 오버피팅의 가능성이 줄고 모델 파라미터 갯수가 훨씬 줄어들었다는 점이다.

## GoogLeNet

- Nin + bock 구조

### Multi Branch Network

- steam (data ingest)
    - given by the first two or three convolutions that operate on the image
    - extract low-level features from the underlying images
- body (data processing)
    - followed by a _body_ of convolutional blocks
- head (prediction)
    - maps the features obtained so far to the required classification, segmentation, detection, or tracking problem at hand

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/2179bfad-7e05-49ed-a9fb-aea0407c9ea0/Untitled.png)

1. 1*1
2. 1_1 → padding =1 3_3
3. 1_1 → padding =2 5_5
4. padding1 3_3 Maxpool → 1_1

![Screenshot 2024-02-07 at 1.23.00 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/cf7e79fc-fcd2-41f8-8fb6-cdae83698cc8/Screenshot_2024-02-07_at_1.23.00_PM.png)

![Screenshot 2024-02-07 at 1.23.48 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/9f2adcc7-9406-4e19-b20c-8d1fa411f7cf/Screenshot_2024-02-07_at_1.23.48_PM.png)

1. 7*7 CONV
2. 3*3 MaxPool
3. 1*1 CONV
4. 3*3 CONV
5. 3*3 MaxPool
6. Inception * 2
7. 3*3 MaxPool
8. Inception *5
9. 3*3 Maxpool
10. Inception *2
11. Global AvgPool
12. FC

## Batch Normalization

Batch Norm VS Layer Norm

- 각 채널 단위로 정규화
- 각 관측치 단위로 정규화

## ResNet & ResNeXt

- Gradient Vanishing problem

### Function Class

### Residual Block

![Screenshot 2024-02-07 at 1.30.14 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/232424d7-69e9-4141-a890-7c036405004a/Screenshot_2024-02-07_at_1.30.14_PM.png)

- Input : x
- function : f(x)

- 왼쪽의 기존모델은 블록의 출력값이 바로 _f_(_**x**_) 인데 반해 오른쪽 모델은 합성곱 연산을 통해 얻은 결과 _g_(_**x**_) 에 기존 입력값 _**x**_ 를 더한 _f_(_**x**_)=_g_(_**x**_)+_**x**_ 를 블록의 출력값으로 하고 있다.
- 이는 기존의 학습한 정보_**x**_ 를 보존하고, 거기에 추가로 학습_g_(_**x**_) 을 하게 되는 방식으로 이해할 수 있다. 이 _g_(_**x**_) 를 잔차 residual 이라고 부른다
- 이는 레이어가 깊어져 많이 학습될수록 _**x**_ 는 점점 출력값 _f_(_**x**_) 에 가까워져 추가학습량 _g_(_**x**_)=_f_(_**x**_)−_x_→0 이 되어야 한다는 의미이다.따라서 학습의 목표는 _g_(_**x**_)=_f_(_**x**_)−_x_→0 로, residual을 0으로 가깝게 만드는 것이 목표가 된다
- 이 방법은 역전파하게 되었을 때 _f_(_**x**_) 를 미분하게 된다. 이는 _g_(_**x**_)+_**x**_ 를 미분하는 것인데, 이때 아무리 미분을 해도 **1** 은 남기 때문에 기울기 소실 문제를 예방할 수 있다

![Screenshot 2024-02-07 at 1.30.37 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/179666b1-a780-4ee4-b908-a6e1b74e8879/Screenshot_2024-02-07_at_1.30.37_PM.png)

- 첫번째에서 2개의 레이어는 GoogLeNet과 동일한 구조이다. 다만 중간사이 배치 정규화가 있다는 차이점은 존재한다
- 그 다음부터는 GoogLeNet이 4개의 인셉션 모듈을 사용한것과는 달리, ResNet은 residual block을 사용했다는 점에서 차이가 있다.

### ResNeXt

- One of the challenges one encounters in the design of ResNet is the trade-off between nonlinearity and dimensionality within a given block.

![Screenshot 2024-02-07 at 1.30.55 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/c6d64075-9b75-46a7-a8ff-ef83ad8356fb/Screenshot_2024-02-07_at_1.30.55_PM.png)

## DenseNet

![Screenshot 2024-02-07 at 1.33.50 PM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/56097ac2-9c18-4918-a18d-a7b0dec27904/5e89e9cf-eeae-4b3f-a284-da8cc7d39516/Screenshot_2024-02-07_at_1.33.50_PM.png)

ResNet에 대해 다시 살펴보면 아래 구조와 같이 liinear한 x와 nonlinear한 g(x)를 더한, f(x) = x + g(x) 로 하나의 block의 연산이 이뤄진다. 이 때 DenseNet은 두 term을 더하는 것이 아닌 concatenate한다는 것에서 차이점이 있다.