

# Evolution of MobileNets and ShuffleNets

[TOC]

MobileNets are the benchmarks in lightweight networks.  They have relatively small sizes (fewer parameters) and comparable efficiencies, yet they don't have very complicated combinations across blocks.

Besides, we will also cover ShuffleNets, which bring many creative and valuable ideas to this field.

## MobileNetV1

In 2017, Google proposed the first generation MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). It introduces a new type of convolution named  ***depthwise separable convolution***. 

![img](./Evolution of MobileNets (and ShuffleNets).assets/v2-e6ef5e7b681a549831d98d094fb3d1c0_720w.jpg?raw=true)

As we can see in the figure, its computational cost: 
$$
DepthwiseSeparableConv=DepthwiseConv+ PointwiseConv \\
= (N*N*K*K*C_{in}) + (N*N*C_{in}*C_{out})
$$
compared with standard convolutions': 
$$
StandardConv = N*N*K*K*C_{in}* C_{out} \\
$$
and the ratio is:
$$
Ratio=  C_{out} + K * K
$$


Normally $K=3\ or \ 5$, but $C_{out}$ could be much bigger, a significant speedup in convolutions!  

------------

Enhanced with this, we can build rather efficient blocks and stack them together. In addition,  two useful hyper-parameters were also introduced:  *Width Multiplier* $\alpha$ and *Resolution Multiplier* $\rho$. The former mainly control the number of output channels, and the latter is to reduce input images' sizes (224, 192, 160...).

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226225133725.png" alt="image-20200226225133725" style="zoom: 67%;" />

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200226233846220.png" alt="image-20200226233846220" style="zoom: 67%;" />



## ShuffleNetV1


A few months later, Face++ introduced another efficient network [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), which employs two novel operations, ***channel shuffle*** and  ***pointwise group convolution***, and it  outperforms MobileNetV1 on ImageNet classification task.

The metric issue must be emphasized. At that time, we were using indirect metrics ($MFLOPs$ or $MAdds$) to measure computational cost, instead of the direct metric $speed$.  As you can imagine, some operations' time may be <u>underestimated</u>, like *Channel Shuffle* (extra overheads in memory copy). 

------------------

### Channel Shuffle for Group Convolutions

![1582796198030](./Evolution of MobileNets (and ShuffleNets).assets/1582796198030.png)

For *group convolution*, the authors split channels into different groups, then apply convolutions separately. *Channel shuffle* is the bridge across convolution groups,  making it possible to utilize information from other groups (hmmm... just sounds reasonable).

-----------

### ShuffleNetV1 Units

![1582798068150](./Evolution of MobileNets (and ShuffleNets).assets/1582798068150.png)

There are two types of units for different $stride(=1\  or\  2)$, and they are quite similar to **ResNet units**. We can stack them to be an efficient network.

---------------

![1582798916272](./Evolution of MobileNets (and ShuffleNets).assets/1582798916272.png)

We should also note that *Channel Shuffle* is not a built-in operation in many frameworks, so you may need to combine *slice* and *gather* operations to mimic it, and be aware of its cost. 



## MobileNetV2

In January 2018,  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)  was proposed.  The key feature is applying **Inverted Residuals** where the shortcut connections are between the thin bottleneck layers.  Within this structure,  the intermediate expansion layer uses lightweight depthwise convolutions followed by non-linearity activations. Meanwhile, they found <u>non-linearities should not be used in the narrow layers.</u>

----------

### Non-linearities in narrow layers

![1582877676954](./Evolution of MobileNets (and ShuffleNets).assets/1582877676954.png)

Why shouldn't we use non-linearities in narrow layers?  Let's discuss *ReLU* to get some intuitions.  

1.  *ReLU* sets all negative parts to $0$,  which definitely  lose some source information.
2. In wide layers,  they should have redundant  information from sources,  so the removal of negative parts won't hurt them too much.
3. As the above figure shows, once a 2D spiral is projected and  activated by *ReLU*, we can hardly reconstruct it from low-dimensional spaces.

------------

### Inverted Residuals

The classical *residual blocks* use bottleneck layers to compress input features to compact features (may contain all the necessary information),  while *inverted residuals* use expansion layers to produce higher-dimensional features (*no activations*, no hurt), then use *depthwide convolution (activation = ReLU)*, to prevent *ReLU* from destroying too much input's information. According to the paper, the expansion factor $t=6$ except in the first bottleneck.

![1582881738599](./Evolution of MobileNets (and ShuffleNets).assets/1582881738599.png)

Still, the block structures: 

![1582880723148](./Evolution of MobileNets (and ShuffleNets).assets/1582880723148.png)

------------

![1582880930157](./Evolution of MobileNets (and ShuffleNets).assets/1582880930157.png)



## ShuffleNetV2

In the middle of 2018, Face++ and THU upgraded ShuffleNetV1 to [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). The paper is highly recommended, which derives several practical and detailed guidelines for efficient network architecture design and shows convincing experimental results.

Prior to ShuffleNetV2, most works only consider $FLOPs$ or $MAdds$. However, in engineering,  ***memory access cost (MAC)***  and ***degree of parallelism*** also contribute to our final inference time. For example, in (c) and (d) networks with similar $FLOPs$ have different $speeds$.

![image-20200229122751852](Evolution of MobileNets (and ShuffleNets).assets/image-20200229122751852.png)

Let's set up our parameters before discussing guidelines. $c_1 = $ the number of input channels, $c_2 = $ the number of input channels,  $h=$ the height of the feature map, $w$ = the width of the feature map.

----------

### G1) Equal channel width minimizes memory access cost (MAC)

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
&\ge hw*2 \sqrt{c_1c_2} + c_1c_2 \\
&= 2 \sqrt{hwB} + \frac{B}{hw}\\
\end{align}
$$
and therefore given fixed $B$, $MAC$ reaches the minimum when $c_1 = c_2$.

![image-20200229154601324](Evolution of MobileNets (and ShuffleNets).assets/image-20200229154601324.png)

-------

### G2) Excessive group convolution increases MAC

*Group convolution* is to reduce $FLOPs$ in convoluting feature maps (e.g., *depthwise convolution*). But when given a fixed $FLOPs$, as the number of groups $g$ going up,  $MAC$ increases.

Now we focus on $1 \times 1$ *group convolution*,  
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

-----
### G3) Network fragmentation reduces degree of parallelism

This is easily understandable. **Multi-path** is a common technique to generate different sets of features and improve accuracy. On the other hand, compared with "one big operation", using "one set of separating operations" will surely introduce extra overheads such as synchronization, like *group convolution*. It may slow down strong parallel computing devices like GPU.

![image-20200229170702941](Evolution of MobileNets (and ShuffleNets).assets/image-20200229170702941.png)

---

### G4) Element-wise operations are non-negligible

 The element-wise operators have small $FLOPs$ but relatively heavy $MAC$, e.g., *ReLU*, *AddTensor*, *AddBias*. Specially, the authors also consider *depthwise convolution* as an element-wise operator because of the  high $MAC/FLOPs$ ratio.

![image-20200229171253171](Evolution of MobileNets (and ShuffleNets).assets/image-20200229171253171.png)

### Conclusion

1. Use "balanced" convolutions (equal channel width); Base case: ShuffleNetV1 and MobileNetV2
2. Be aware of the cost of using group convolution; Base case: ShuffleNetV1
3. Reduce the degree of fragmentation; Base case: Inception Series and ShuffleNetV1
4. Reduce element-wise operations; Base case: MobileNetV2

---

Let's consider building units. The guidelines are only beneficial for efficiency, but the capacity of models is equally important. ShuffleNetV1 has been proven as an efficient model. Therefore, we use it as the base model,  and try to improve it within our guidelines. 

1. To Follow G1,  ShuffleNetV1 has bottlenecks (such as 256-d -> 64-d -> 256-d), so we remove these bottlenecks by changing all $1 \times 1 \ GConv (c_{in} \ne c_{out}) $ to $1 \times 1 \ GConv \ (c_{in} = c_{out})$. 
2. To Follow G2, we further change $1 \times 1 \ GConv \ (c_{in} = c_{out})$ to  $1 \times 1 \ Conv \ (c_{in} = c_{out})$, making *channel shuffle* will not exchange extra information, thus move it to the bottom.
3. To Follow G3, the raw $1 \times 1 \ GConv$ should be blamed again. Meanwhile, we shall keep the number of branches low ($=2$).
4. To Follow G4, replace *add* with *concat* while merging two branches.

Now, we are facing a contradiction that the number of output channels is larger than that of input channels in the units. To fix it, a simple operator called *channel split* was introduced, which equally split input channels into two branches.  Nevertheless, while doing spatial downsampling, *channel split* is removed in order to double the number of output channels.

Also, the three successive element-wise operations, *concat*, *channel shuffle* and *channel split* , can be  merged into a single element-wise operation. 

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200229173424082.png" alt="image-20200229173424082" style="zoom:67%;" />

<img src="Evolution of MobileNets (and ShuffleNets).assets/image-20200229173955686.png" alt="image-20200229173955686" style="zoom:67%;" />




## MobileNetV3
Time flies to June 2019. The most cutting-edge and GPU-consuming technology, **network architecture search (NAS)** has been being applied widely, and gradually reducing job positions in the DL market. This generation of MobileNets [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) is tuned to mobile phone CPUs. 

![image-20200304155730379](Evolution of MobileNets (and ShuffleNets).assets/image-20200304155730379.png)

---

### Block-wise search

First, For block-wise search in **Large** mobile model, we employ a hardware-aware NAS approach which is similar to **MnasNet**, and get similar results. Therefore, we simply reuse the same **MnasNet-A1** as our initial Large model.

Then, for **Small** mobile model, we change multi-objective reward as follow:
$$
ACC(m) \times {[LAT(T) / TAR]}^w
$$
where $ACC(m)$ is accuracy, $LAT(m)$ is model's latency, and $TAR$ stands for target latency. Specially, $w=-0.15$ (vs $w=-0.07$ in MnasNet) to  compensate for the larger accuracy change for different latencies.  Upon this, we apply NetAdapt and other optimizations to obtain the final **MobileNetV3-Small** model.

-----------------

### Layer-wise search

For layer-wise search, we modify this **NetAdapt** algorithm in MnasNet and minimize the ratio between latency change and accuracy change $(\frac{\Delta Acc}{\Delta latency})$.  In short the technique proceeds as follows:

> 1. Starts with a seed network architecture found by platform-aware NAS.
>
> 2. For each step:
>
>   (a) Generate a set of new *proposals*. Each proposal represents a modification of an architecture that generates at least  $\delta=0.01|L|$($L$ is the latency of the seed model) reduction in latency compared to the previous step.
>
>   (b) For each proposal we use the pre-trained model from the previous step and populate the new proposed architecture, truncating and randomly initializing missing weights as appropriate. Finetune each proposal for $T=1000$ steps to get a coarse estimate of the accuracy.
>
>   (c) Selected best proposal according to some metric.
>
> 3. Iterate previous steps until target latency is reached.

-----------

### Redesigning Expensive Layers and Nonlinearities

![image-20200304143534725](Evolution of MobileNets (and ShuffleNets).assets/image-20200304143534725.png)

The original last stage uses *inverted residual* to expand features, followed by *average pooling* and $1 \times 1$ convolution (dense layer), to get the final prediction.  Now, we move *average pooling* forward, and the final set of features is now computed at $1 \times 1$ spatial resolution instead of $7 \times 7$ spatial resolution. This approach reduces 7ms latency and almost no hurt in accuracy.

Beside, we replace the filter number of the first $3 \times 3$ convolution layer (to produce initial features from image) from 32 to 16.

***Swish*** is an effective substitute for *ReLU*, which can improve accuracy in many neural networks. However, the sigmoid part in *swish* is quite expensive to compute.
$$
swish(x) = x*\sigma(x)
$$
We replace the original sigmoid function to hard-sigmoid: $\frac{ReLU6(x+3)}{6}$, then we have:
$$
h\mbox{-}swish(x) = x*\frac{ReLU6(x+3)}{6}
$$
![image-20200304153544169](Evolution of MobileNets (and ShuffleNets).assets/image-20200304153544169.png)

-------

### Large squeeze-and-excite

 We set all the size of the **squeeze-and-excite** bottleneck to fixed to be $1/4$ of the number of channels in expansion layer, without discernible latency cost. We should also note that the *FCs* can be replaced by convolution layers.

![image-20200304154452538](Evolution of MobileNets (and ShuffleNets).assets/image-20200304154452538.png)

----------------------

The final MobileNetV3-Large and MobileNetV3-Small model:

![image-20200304154512410](Evolution of MobileNets (and ShuffleNets).assets/image-20200304154512410.png)

![image-20200304154520838](Evolution of MobileNets (and ShuffleNets).assets/image-20200304154520838.png)