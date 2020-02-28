# Evolution of MobileNets (and ShuffleNets?)

[TOC]

MobileNets are the benchmarks in lightweight networks.  They have relative small sizes (less parameters than others) and comparable efficiencies, yet they don't have too much complicated combinations across blocks.

Besides, we will also cover ShuffleNets, which bring many creative and valuable ideas to this area.

## MobileNetV1

In 2017, Google proposed the first generation MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). It introduces a new type of convolution, namely  ***Depthwise Separable Convolution***. 

![img](Evolution of MobileNets (and ShuffleNets).assets/v2-e6ef5e7b681a549831d98d094fb3d1c0_720w.jpg)

As we can see from the figure, its computation cost equals: 
$$
DepthwiseSeparableConv=DepthwiseConv+ PointwiseConv \\
= (N*N*K*K*C_{in}) + (N*N*C_{in}*C_{out})
$$
compared with ***Standard Convolution***'s': 
$$
StandardConv = N*N*K*K*C_{in}* C_{out} \\
$$
and the ratio is:
$$
Ratio=  C_{out} + K * K
$$
Normally $K=3\ or \ 5$, but $C_{out}$ could be much bigger, a significant speedup in convolutions!  With this, we can build rather efficient blocks and stack them to be **MobileNetV1**.

![image-20200226225133725](Evolution of MobileNets (and ShuffleNets).assets/image-20200226225133725.png)

![image-20200226233846220](Evolution of MobileNets (and ShuffleNets).assets/image-20200226233846220.png)

In addition,  two useful hyperparameters were also introduced:  ***Width Multiplier* $\alpha$** and ***Resolution Multiplier* $\rho$**. The former mainly control ***the number of output channels***, and the latter is to reduce ***input images' sizes*** (224, 192, 160...).



## ShuffleNetV1


A few months later, Face++ introduced another efficient network [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), which employs two novel operations, ***Channel Shuffle*** and  ***Pointwise Group Convolution***, and it  outperforms **MobileNetV1** on *ImageNet classification task*.

The metric issue must be emphasized. At that time, we were using proxy metrics (e.g. ***MFLOP, MAdd***) to measure computation cost, not using direct metric ***Speed***.  As you can imagine, some operations' costs may be **<u>underestimated</u>**, like ***Channel Shuffle*** (extra overheads in memory copy) . 

![1582796198030](Evolution of MobileNets (and ShuffleNets).assets/1582796198030.png)

For ***Group Gonvolution***, the authors split channels into different groups, then do convolutions separately. ***Channel Shuffle*** is the bridge across convolution groups,  making it possible to utilize information from other groups (emmm... just sounds reasonable).

![1582798068150](Evolution of MobileNets (and ShuffleNets).assets/1582798068150.png)

There are two types of Units for different ***Stride*** (b=1 or c=2), and they are quite similar to **ResNet Units**. We can stack them to be an efficient network.

![1582798916272](Evolution of MobileNets (and ShuffleNets).assets/1582798916272.png)

We should also note that ***Channel Shuffle*** is not a built-in operation in many frameworks, so you may need to combine ***Slice*** and ***Gather*** operations to mimic it, and be aware of its cost. 



## MobileNetV2

In January 2018,  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  was proposed.  The key feature is applying **Inverted Residuals** where the shortcut connections are between the thin bottleneck layers.  Within this structure,  the intermediate expansion layer uses lightweight depthwise convolutions followed by non-linearity activations. Meanwhile, they found <u>non-linearities should not be used in the narrow layers.</u>

![1582877676954](Evolution of MobileNets (and ShuffleNets).assets/1582877676954.png)

Why shouldn't use non-linearities in narrow layers?  Let's discuss *ReLU* to get some intuitions.  

1.  *ReLU* sets all negative parts to $0$,  which definitely  loss some source information.
2. In wide layers,  they should have redundant  information from sources,  so the removal of negative parts won't hurt them too much.
3. As the above figure shows, once a 2D spiral is projected and  activated by *ReLU* , we can hardly reconstruct a from low-dimensional spaces.

**Inverted Residuals** are quite unusual. The classical **Residual blocks** use bottleneck layers to capture compact features (may contain all the necessary information),  while **Inverted Residuals** use expansion layers to produce higher-dimensional features (*linear activations*), then use **Depthwide Convolutions** followed *ReLU*, to prevent *ReLU* from destroying too much information.

![1582881738599](Evolution of MobileNets (and ShuffleNets).assets/1582881738599.png)

Still, the structures: 

![1582880723148](Evolution of MobileNets (and ShuffleNets).assets/1582880723148.png)

![1582880930157](Evolution of MobileNets (and ShuffleNets).assets/1582880930157.png)

## ShuffleNetV2

[ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)



## MobileNetV3

[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

