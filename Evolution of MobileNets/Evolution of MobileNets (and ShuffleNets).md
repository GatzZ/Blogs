# Evolution of MobileNets (and ShuffleNets?)

MobileNets are the benchmarks in lightweight networks.  They have relative small sizes (less parameters than others) and comparable efficiencies, yet they don't have too much complicated combinations across blocks.

Besides, we will talk about ShuffleNets, which bring many creative and valuable ideas to this area.

## MobileNetV1

In 2017, Google proposed the first generation MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). It introduces a new type of convolution, namely  Separable Convolution**. 



| <img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226225133725.png" alt="image-20200226225133725" style="zoom: 50%;" /><img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226233846220.png" alt="image-20200226233846220" style="zoom: 50%;" /> |      |
| :----------------------------------------------------------: | ---- |
|                                                              |      |

You can easily see the structure difference in the left figure,  with computation costs: 
$$
StandardConv = N*N*K*K*C_{in}* C_{out} \\
DepthwiseSeparableConv=(N*N*K*K*C_{in}) + (N*N*C_{in}*C_{out}) \\
$$
and the ratio is:
$$
Ratio=  C_{out} + K * K
$$
Normally $K=3\ or \ 5$, but $C_{out}$ could be much bigger, a significant speedup in convolutions!  

In addition,  two useful hyperparameters were also introduced:  ***Width Multiplier* $\alpha$** and ***Resolution Multiplier* $\rho$**. The former mainly control ***the number of output channels***, and the latter is to reduce ***input images' sizes*** (224, 192, 160...).

## ShuffleNetV1

A few months later, Face++ introduced another efficient network **ShuffleNet**, which employs two novel operations, ***channel shuffle*** and  ***pointwise group convolution***, and it  outperformed **MobileNetV1** on *ImageNet classification task*.

The metric issue must be emphasized. At that time, we were using a proxy metric ***MFLOPS*** to measure computation cost, not using direct metric ***Speed***.  As you can imagine, some operations' costs may be **<u>underestimated</u>**, like ***channel shuffle*** (extra overheads in memory copy) . 

![1582796198030](Evolution of MobileNets (and ShuffleNets).assets/1582796198030.png)

For ***group convolution***, the authors split channels into different groups, then do convolutions separately. ***Channel shuffle*** is the bridge across convolution groups,  making it possible to utilize information from other groups (emmm... just sounds reasonable).

![1582798068150](Evolution of MobileNets (and ShuffleNets).assets/1582798068150.png)

There are two types of Units for different ***stride*** (b=1 or c=2), and they are quite similar to **ResNet Units**. We can stack them to be an efficient network.

![1582798916272](Evolution of MobileNets (and ShuffleNets).assets/1582798916272.png)

We should also note that ***channel shuffle*** is not a built-in operation in many frameworks, so you may need to combine ***slice*** and ***gather*** operations to mimic it, and be aware of its cost. 