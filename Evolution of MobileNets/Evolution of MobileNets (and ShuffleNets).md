# Evolution of MobileNets (and ShuffleNets?)

[TOC]

MobileNets are the benchmarks in lightweight networks.  They have relative small sizes (less parameters than others) and comparable efficiencies, yet they don't have too much complicated combinations across blocks.

Besides, we will also cover ShuffleNets, which bring many creative and valuable ideas to this area.

## MobileNetV1

In 2017, Google proposed the first generation MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). It introduces a new type of convolution, namely  ***Depthwise Separable Convolution***. 

![img](./Evolution of MobileNets (and ShuffleNets).assets/v2-e6ef5e7b681a549831d98d094fb3d1c0_720w.jpg?raw=true)

As we can see from the figure, its a cost equals: 
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

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226225133725.png" alt="image-20200226225133725" style="zoom: 67%;" />

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226233846220.png" alt="image-20200226233846220" style="zoom: 67%;" />

In addition,  two useful hyperparameters were also introduced:  ***Width Multiplier* $\alpha$** and ***Resolution Multiplier* $\rho$**. The former mainly control ***the number of output channels***, and the latter is to reduce ***input images' sizes*** (224, 192, 160...).



## ShuffleNetV1


A few months later, Face++ introduced another efficient network [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), which employs two novel operations, ***Channel Shuffle*** and  ***Pointwise Group Convolution***, and it  outperforms **MobileNetV1** on *ImageNet classification task*.

The metric issue must be emphasized. At that time, we were using indiect metrics (e.g. ***MFLOPs, MAdds***) to measure computational cost, not using direct metric ***Speed***.  As you can imagine, some operations' costs may be **<u>underestimated</u>**, like ***Channel Shuffle*** (extra overheads in memory copy) . 

![1582796198030](./Evolution of MobileNets (and ShuffleNets).assets/1582796198030.png)

For ***Group Gonvolution***, the authors split channels into different groups, then do convolutions separately. ***Channel Shuffle*** is the bridge across convolution groups,  making it possible to utilize information from other groups (emmm... just sounds reasonable).

![1582798068150](./Evolution of MobileNets (and ShuffleNets).assets/1582798068150.png)

There are two types of Units for different ***Stride*** (b=1 or c=2), and they are quite similar to **ResNet Units**. We can stack them to be an efficient network.

![1582798916272](./Evolution of MobileNets (and ShuffleNets).assets/1582798916272.png)

We should also note that ***Channel Shuffle*** is not a built-in operation in many frameworks, so you may need to combine ***Slice*** and ***Gather*** operations to mimic it, and be aware of its cost. 



## MobileNetV2

In January 2018,  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  was proposed.  The key feature is applying **Inverted Residuals** where the shortcut connections are between the thin bottleneck layers.  Within this structure,  the intermediate expansion layer uses lightweight depthwise convolutions followed by non-linearity activations. Meanwhile, they found <u>non-linearities should not be used in the narrow layers.</u>

![1582877676954](./Evolution of MobileNets (and ShuffleNets).assets/1582877676954.png)

Why shouldn't use non-linearities in narrow layers?  Let's discuss *ReLU* to get some intuitions.  

1.  *ReLU* sets all negative parts to $0$,  which definitely  loss some source information.
2. In wide layers,  they should have redundant  information from sources,  so the removal of negative parts won't hurt them too much.
3. As the above figure shows, once a 2D spiral is projected and  activated by *ReLU* , we can hardly reconstruct it from low-dimensional spaces.

The classical **Residual blocks** use bottleneck layers to capture compact features (may contain all the necessary information),  while **Inverted Residuals** use expansion layers to produce higher-dimensional features (*no activations*, no hurt), then use **Depthwide Convolutions** followed *ReLU*, to prevent *ReLU* from destroying too much source's information. According to the paper, the expansion factor $t=6$ except in the first bottleneck.

![1582881738599](./Evolution of MobileNets (and ShuffleNets).assets/1582881738599.png)

Still, the structures: 

![1582880723148](./Evolution of MobileNets (and ShuffleNets).assets/1582880723148.png)

![1582880930157](./Evolution of MobileNets (and ShuffleNets).assets/1582880930157.png)

## ShuffleNetV2

In the middle of 2018, Face++ and THU upgraded ShuffleNetV1 to [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). The paper is very detailed and practical, which derive several guidelines for efficient network architecture design and shows impressive experimental results.

Prior to ShuffleNetV2, most works only consider $FLOPs$ or $MAdds$. However, in engineering,  ***memory access cost (MAC)***  and ***degree of parallelism*** also contribute to our final inference time. For example, in (c) and (d) networks with similar $FLOPs$ have different $speeds$.

![image-20200229122751852](Evolution of MobileNets (and ShuffleNets).assets/image-20200229122751852.png)

Let's set up our parameters before discussing guidelines. $c_1 = $ the number of input channels, $c_2 = $ the number of input channels,  $h=$ height of the feature map, $w$ = width of the feature map.

--------
> **G1) Equal channel width minimizes memory access cost (MAC)**

For simplicity, we just consider the $1 \times 1$ convolution. The computational cost ($FLOPs$): 
$$
B = hwc_1c_2
$$
and memory access cost:
$$
\begin{align}
MAC &= MAC_{in} + MAC_{out} + MAC_{kernel} \\
&= hwc_1 + hwc_2 + c_1c_2 \\
\end{align}
$$
From *Inequality of arithmetic and geometric means*:
$$
x+y \ge 2\sqrt{xy}
$$
we have:
$$
\begin{align}
MAC &= hwc_1 + hwc_2 + c_1c_2 \\
&\ge 2 \sqrt{h^2w^2c_1c_2} + c_1c_2 \\
&= 2 \sqrt{hwB} + \frac{B}{hw}\\
\end{align}
$$
and therefore given certain $B$, $MAC$ reaches the minimum when $c_1 = c_2$.

![image-20200229154601324](Evolution of MobileNets (and ShuffleNets).assets/image-20200229154601324.png)

---------

> **G2) Excessive group convolution increases MAC**

**Group convolution** is  a common way to reduce $FLOPs$ while convoluting feature maps (e.g., **depthwise convolution**). But when given a certain $FLOPs$, as the number of groups $g$ going up,  $MAC$ increases.

Now we focus on $1 \times 1$ **group convolution**,  
$$
B = \frac{hwc_1c_2}{g}
$$
and
$$
\begin{align}
MAC &= hwc_1 + hwc_2 + \frac{c_1c_2}{g} \\
&= hwc_1 + \frac{Bg}{c_1} + \frac{B}{hw} \\
\end{align}
$$
![image-20200229162743298](Evolution of MobileNets (and ShuffleNets).assets/image-20200229162743298.png)
---
> **G3) Network fragmentation reduces degree of parallelism**

This is so understandable. **Multi-path** is a common technique to generate different types of features and improve accuracy. On the other hand, compared with "one big operation", using "one set of separating operations" will surely introduce extra overheads such as synchronization, like **group convolution**. It may slow down strong parallel computing devices like GPUs.

![image-20200229170702941](Evolution of MobileNets (and ShuffleNets).assets/image-20200229170702941.png)

---

> **G4) Element-wise operations are non-negligible**

 The element-wise operators have small FLOPs but relatively heavy MAC, e.g., *ReLU*, *AddTensor*, *AddBias*. Specially, the authors also consider **depthwise convolution** as a element-wise operator because of the  high $MAC/FLOPs$ ratio.

![image-20200229171253171](Evolution of MobileNets (and ShuffleNets).assets/image-20200229171253171.png)

> **Conclusion**
1. Use "balanced" convolutions (equal channel width); Base case: ShuffleNetV1 and MobileNetV2
2. Be aware of the cost of using group convolution; Base case: ShuffleNetV1
3. Reduce the degree of fragmentation; Base case: Inception Series and ShuffleNetV1
4. Reduce element-wise operations; Base case: MobileNetV2

---

Let's consider building units. The guidelines are only beneficial for efficiency, but the capacity of models is equally important. **ShuffleNetV1** has been proven as a efficient model. Therefore, we use it as the base-model,  and try to improve it within our guidelines. 

1. Following G1,  **ShuffleNetV1** has **bottlenecks** (such as 256-d -> 64-d -> 256-d) in right branches, so we remove these bottlenecks by changing $1 \times 1 \ GConv \ (c_{in} \ne c_{out}) $ to $1 \times 1 \ GConv \ (c_{in} = c_{out})$. 
2. Following G2, we change $1 \times 1 \ GConv$ to $1 \times 1 \ Conv$, then **channel shuffle** in the right branch will not exchange extra information, thus move it to the bottom.
3. Following G3, $1 \times 1 \ GConv$ should also be blamed and keep the number of branches low ($=2$).
4. Following G4, replace *Add*​ with *Concat* while merging the two branches.

Now, we are facing a contradiction that the number of output channels is larger than that of input channels in the units. To fix it, a simple operator called **channel split** was introduced, which equally split input channels to two branches.  Nevertheless, while doing spatial down sampling, **channel split** is removed, cause we need to double the number of output channels.

Also, the three successive element-wise operations, **Concat, Channel Shuffle and Channel Split** , can be  merged into a single element-wise operation. 

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200229173424082.png" alt="image-20200229173424082" style="zoom:67%;" />

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200229173955686.png" alt="image-20200229173955686" style="zoom:67%;" />

## MobileNetV3

[Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)

