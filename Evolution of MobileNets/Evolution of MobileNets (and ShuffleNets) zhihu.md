

# Evolution of MobileNets and ShuffleNets

MobileNets are the benchmarks in lightweight networks.  They have relatively small sizes (fewer parameters) and comparable efficiencies, yet they don't have very complicated combinations across blocks.

Besides, we will also cover ShuffleNets, which bring many creative and valuable ideas to this field.

## MobileNetV1

In 2017, Google proposed the first generation MobileNet: [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). It introduces a new type of convolution named ***depthwise separable convolution***. 

![img](https://tva1.sinaimg.cn/large/00831rSTgy1gcla8t2n4wj30go082t8z.jpg)

As we can see in the figure, its computational cost: 

<img src="https://www.zhihu.com/equation?tex=DepthwiseSeparableConv=DepthwiseConv+ PointwiseConv \\
= (N*N*K*K*C_{in}) + (N*N*C_{in}*C_{out})" alt="DepthwiseSeparableConv=DepthwiseConv+ PointwiseConv \\
= (N*N*K*K*C_{in}) + (N*N*C_{in}*C_{out})" class="ee_img tr_noresize" eeimg="1">

compared with standard convolutions': 

<img src="https://www.zhihu.com/equation?tex=StandardConv = N*N*K*K*C_{in}* C_{out} \\" alt="StandardConv = N*N*K*K*C_{in}* C_{out} \\" class="ee_img tr_noresize" eeimg="1">

and the ratio is:

<img src="https://www.zhihu.com/equation?tex=Ratio=  C_{out} + K * K" alt="Ratio=  C_{out} + K * K" class="ee_img tr_noresize" eeimg="1">



Normally 
<img src="https://www.zhihu.com/equation?tex=K=3\ or \ 5" alt="K=3\ or \ 5" class="ee_img tr_noresize" eeimg="1">
, but 
<img src="https://www.zhihu.com/equation?tex=C_{out}" alt="C_{out}" class="ee_img tr_noresize" eeimg="1">
 could be much bigger, a significant speedup in convolutions!  



Enhanced with this, we can build rather efficient blocks and stack them together. In addition,  two useful hyper-parameters were also introduced:  *Width Multiplier* 
<img src="https://www.zhihu.com/equation?tex=\alpha" alt="\alpha" class="ee_img tr_noresize" eeimg="1">
 and *Resolution Multiplier* 
<img src="https://www.zhihu.com/equation?tex=\rho" alt="\rho" class="ee_img tr_noresize" eeimg="1">
. The former mainly control the number of output channels, and the latter is to reduce input images' sizes (224, 192, 160...).

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gcla9qzemlj30hi0bsq4h.jpg" alt="image-20200226225133725" style="zoom: 67%;" />

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gcla9ytlb5j30hs0jon06.jpg" alt="image-20200226233846220" style="zoom: 67%;" />



## ShuffleNetV1


A few months later, Face++ introduced another efficient network [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083), which employs two novel operations, ***channel shuffle*** and ***pointwise group convolution***, and it  outperforms MobileNetV1 on ImageNet classification task.

The metric issue must be emphasized. At that time, we were using indirect metrics (
<img src="https://www.zhihu.com/equation?tex=MFLOPs" alt="MFLOPs" class="ee_img tr_noresize" eeimg="1">
 or 
<img src="https://www.zhihu.com/equation?tex=MAdds" alt="MAdds" class="ee_img tr_noresize" eeimg="1">
) to measure computational cost, instead of the direct metric 
<img src="https://www.zhihu.com/equation?tex=speed" alt="speed" class="ee_img tr_noresize" eeimg="1">
.  As you can imagine, some operations' time may be <u>underestimated</u>, like *Channel Shuffle* (extra overheads in memory copy). 



### Channel Shuffle for Group Convolutions

![1582796198030](https://tva1.sinaimg.cn/large/00831rSTgy1gcladj510jj30lw09mjt2.jpg)

For *group convolution*, the authors split channels into different groups, then apply convolutions separately. *Channel shuffle* is the bridge across convolution groups,  making it possible to utilize information from other groups (hmmm... just sounds reasonable).



### ShuffleNetV1 Units

![1582798068150](https://tva1.sinaimg.cn/large/00831rSTgy1gcladjo0lrj30lm0a075f.jpg)

There are two types of units for different 
<img src="https://www.zhihu.com/equation?tex=stride(=1\  or\  2)" alt="stride(=1\  or\  2)" class="ee_img tr_noresize" eeimg="1">
, and they are quite similar to **ResNet units**. We can stack them to be an efficient network.



![1582798916272](https://tva1.sinaimg.cn/large/00831rSTgy1gcladfe5r6j30mg09ljtk.jpg)

We should also note that *Channel Shuffle* is not a built-in operation in many frameworks, so you may need to combine *slice* and *gather* operations to mimic it, and be aware of its cost. 



## MobileNetV2

In January 2018,  [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381) was proposed.  The key feature is applying **Inverted Residuals** where the shortcut connections are between the thin bottleneck layers.  Within this structure,  the intermediate expansion layer uses lightweight depthwise convolutions followed by non-linearity activations. Meanwhile, they found <u>non-linearities should not be used in the narrow layers.</u>



### Non-linearities in narrow layers

![1582877676954](https://tva1.sinaimg.cn/large/00831rSTgy1gcladefjagj30bn06xta4.jpg)

Why shouldn't we use non-linearities in narrow layers?  Let's discuss *ReLU* to get some intuitions.

1.  *ReLU* sets all negative parts to 
<img src="https://www.zhihu.com/equation?tex=0" alt="0" class="ee_img tr_noresize" eeimg="1">
,  which definitely lose some source information.
2. In wide layers,  they should have redundant information from sources,  so the removal of negative parts won't hurt them too much.
3. As the above figure shows, once a 2D spiral is projected and activated by *ReLU*, we can hardly reconstruct it from low-dimensional spaces.



### Inverted Residuals

The classical *residual blocks* use bottleneck layers to compress input features to compact features (may contain all the necessary information),  while *inverted residuals* use expansion layers to produce higher-dimensional features (*no activations*, no hurt), then use *depthwise convolution (activation = ReLU)*, to prevent *ReLU* from destroying too much input's information. According to the paper, the expansion factor 
<img src="https://www.zhihu.com/equation?tex=t=6" alt="t=6" class="ee_img tr_noresize" eeimg="1">
 except in the first bottleneck.

![1582881738599](https://tva1.sinaimg.cn/large/00831rSTgy1gcladcuccjj30qz08wq40.jpg)

Still, the block structures: 

![1582880723148](https://tva1.sinaimg.cn/large/00831rSTgy1gcladbkib4j30aw0bmaar.jpg)



![1582880930157](https://tva1.sinaimg.cn/large/00831rSTgy1gcladb6ygcj30cm0dtq58.jpg)



## ShuffleNetV2

In the middle of 2018, Face++ and THU upgraded ShuffleNetV1 to [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164). The paper is highly recommended, which derives several practical and detailed guidelines for efficient network architecture design and shows convincing experimental results.

Prior to ShuffleNetV2, most works only consider 
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
 or 
<img src="https://www.zhihu.com/equation?tex=MAdds" alt="MAdds" class="ee_img tr_noresize" eeimg="1">
. However, in engineering,  ***memory access cost (MAC)***  and ***degree of parallelism*** also contribute to our final inference time. For example, in (c) and (d) networks with similar 
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
 have different 
<img src="https://www.zhihu.com/equation?tex=speeds" alt="speeds" class="ee_img tr_noresize" eeimg="1">
.

![image-20200229122751852](https://tva1.sinaimg.cn/large/00831rSTgy1gcladdnj0lj30y80okwig.jpg)

Let's set up our parameters before discussing guidelines. 
<img src="https://www.zhihu.com/equation?tex=c_1 = " alt="c_1 = " class="ee_img tr_noresize" eeimg="1">
 the number of input channels, 
<img src="https://www.zhihu.com/equation?tex=c_2 = " alt="c_2 = " class="ee_img tr_noresize" eeimg="1">
 the number of input channels,  
<img src="https://www.zhihu.com/equation?tex=h=" alt="h=" class="ee_img tr_noresize" eeimg="1">
 the height of the feature map, 
<img src="https://www.zhihu.com/equation?tex=w" alt="w" class="ee_img tr_noresize" eeimg="1">
 = the width of the feature map.



### G1) Equal channel width minimizes memory access cost (MAC)

For simplicity, we just consider the 
<img src="https://www.zhihu.com/equation?tex=1 \times 1" alt="1 \times 1" class="ee_img tr_noresize" eeimg="1">
 convolution. The computational cost (
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
): 

<img src="https://www.zhihu.com/equation?tex=B = hwc_1c_2" alt="B = hwc_1c_2" class="ee_img tr_noresize" eeimg="1">

and memory access cost:

<img src="https://www.zhihu.com/equation?tex=\begin{align}
MAC &= MAC_{in} + MAC_{out} + MAC_{kernel} \\
&= hwc_1 + hwc_2 + c_1c_2 \\
\end{align}" alt="\begin{align}
MAC &= MAC_{in} + MAC_{out} + MAC_{kernel} \\
&= hwc_1 + hwc_2 + c_1c_2 \\
\end{align}" class="ee_img tr_noresize" eeimg="1">

From *Inequality of arithmetic and geometric means*:

<img src="https://www.zhihu.com/equation?tex=x+y \ge 2\sqrt{xy}" alt="x+y \ge 2\sqrt{xy}" class="ee_img tr_noresize" eeimg="1">

we have:

<img src="https://www.zhihu.com/equation?tex=\begin{align}
MAC &= hwc_1 + hwc_2 + c_1c_2 \\
&\ge hw*2 \sqrt{c_1c_2} + c_1c_2 \\
&= 2 \sqrt{hwB} + \frac{B}{hw}\\
\end{align}" alt="\begin{align}
MAC &= hwc_1 + hwc_2 + c_1c_2 \\
&\ge hw*2 \sqrt{c_1c_2} + c_1c_2 \\
&= 2 \sqrt{hwB} + \frac{B}{hw}\\
\end{align}" class="ee_img tr_noresize" eeimg="1">

and therefore given fixed 
<img src="https://www.zhihu.com/equation?tex=B" alt="B" class="ee_img tr_noresize" eeimg="1">
, 
<img src="https://www.zhihu.com/equation?tex=MAC" alt="MAC" class="ee_img tr_noresize" eeimg="1">
 reaches the minimum when 
<img src="https://www.zhihu.com/equation?tex=c_1 = c_2" alt="c_1 = c_2" class="ee_img tr_noresize" eeimg="1">
.

![image-20200229154601324](https://tva1.sinaimg.cn/large/00831rSTgy1gclb9abpqwj316k0coafj.jpg)



### G2) Excessive group convolution increases MAC

*Group convolution* is to reduce 
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
 in convoluting feature maps (e.g., *depthwise convolution*). But when given a fixed 
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
, as the number of groups 
<img src="https://www.zhihu.com/equation?tex=g" alt="g" class="ee_img tr_noresize" eeimg="1">
 going up,  
<img src="https://www.zhihu.com/equation?tex=MAC" alt="MAC" class="ee_img tr_noresize" eeimg="1">
 increases.

Now we focus on 
<img src="https://www.zhihu.com/equation?tex=1 \times 1" alt="1 \times 1" class="ee_img tr_noresize" eeimg="1">
 *group convolution*,  

<img src="https://www.zhihu.com/equation?tex=B = \frac{hwc_1c_2}{g}" alt="B = \frac{hwc_1c_2}{g}" class="ee_img tr_noresize" eeimg="1">

and

<img src="https://www.zhihu.com/equation?tex=\begin{align}
MAC &= hwc_1 + hwc_2 + \frac{c_1c_2}{g} \\
&= hwc_1 + \frac{Bg}{c_1} + \frac{B}{hw} \\
\end{align}" alt="\begin{align}
MAC &= hwc_1 + hwc_2 + \frac{c_1c_2}{g} \\
&= hwc_1 + \frac{Bg}{c_1} + \frac{B}{hw} \\
\end{align}" class="ee_img tr_noresize" eeimg="1">


![image-20200229162743298](https://tva1.sinaimg.cn/large/00831rSTgy1gcladk2oo5j31600datd7.jpg)


### G3) Network fragmentation reduces degree of parallelism

This is easily understandable. **Multi-path** is a common technique to generate different sets of features and improve accuracy. On the other hand, compared with "one big operation", using "one set of separating operations" will surely introduce extra overheads such as synchronization, like *group convolution*. It may slow down strong parallel computing devices like GPU.

![image-20200229170702941](https://tva1.sinaimg.cn/large/00831rSTgy1gcladg2sc6j316o0ecjwx.jpg)



### G4) Element-wise operations are non-negligible

 The element-wise operators have small 
<img src="https://www.zhihu.com/equation?tex=FLOPs" alt="FLOPs" class="ee_img tr_noresize" eeimg="1">
 but relatively heavy 
<img src="https://www.zhihu.com/equation?tex=MAC" alt="MAC" class="ee_img tr_noresize" eeimg="1">
, e.g., *ReLU*, *AddTensor*, *AddBias*. Specially, the authors also consider *depthwise convolution* as an element-wise operator because of the high 
<img src="https://www.zhihu.com/equation?tex=MAC/FLOPs" alt="MAC/FLOPs" class="ee_img tr_noresize" eeimg="1">
 ratio.

![image-20200229171253171](https://tva1.sinaimg.cn/large/00831rSTgy1gcladgyiahj31680d2n2a.jpg)

### Conclusion

1. Use "balanced" convolutions (equal channel width); Base case: ShuffleNetV1 and MobileNetV2
2. Be aware of the cost of using group convolution; Base case: ShuffleNetV1
3. Reduce the degree of fragmentation; Base case: Inception Series and ShuffleNetV1
4. Reduce element-wise operations; Base case: MobileNetV2



Let's consider building units. The guidelines are only beneficial for efficiency, but the capacity of models is equally important. ShuffleNetV1 has been proven as an efficient model. Therefore, we use it as the base model and try to improve it within our guidelines. 

1. To Follow G1,  ShuffleNetV1 has bottlenecks (such as 256-d -> 64-d -> 256-d), so we remove these bottlenecks by changing all 
<img src="https://www.zhihu.com/equation?tex=1 \times 1 \ GConv (c_{in} \ne c_{out}) " alt="1 \times 1 \ GConv (c_{in} \ne c_{out}) " class="ee_img tr_noresize" eeimg="1">
 to 
<img src="https://www.zhihu.com/equation?tex=1 \times 1 \ GConv \ (c_{in} = c_{out})" alt="1 \times 1 \ GConv \ (c_{in} = c_{out})" class="ee_img tr_noresize" eeimg="1">
. 
2. To Follow G2, we further change 
<img src="https://www.zhihu.com/equation?tex=1 \times 1 \ GConv \ (c_{in} = c_{out})" alt="1 \times 1 \ GConv \ (c_{in} = c_{out})" class="ee_img tr_noresize" eeimg="1">
 to  
<img src="https://www.zhihu.com/equation?tex=1 \times 1 \ Conv \ (c_{in} = c_{out})" alt="1 \times 1 \ Conv \ (c_{in} = c_{out})" class="ee_img tr_noresize" eeimg="1">
, making *channel shuffle* will not exchange extra information, thus move it to the bottom.
3. To Follow G3, the raw 
<img src="https://www.zhihu.com/equation?tex=1 \times 1 \ GConv" alt="1 \times 1 \ GConv" class="ee_img tr_noresize" eeimg="1">
 should be blamed again. Meanwhile, we shall keep the number of branches low (
<img src="https://www.zhihu.com/equation?tex==2" alt="=2" class="ee_img tr_noresize" eeimg="1">
).
4. To Follow G4, replace *add* with *concat* while merging two branches.

Now, we are facing a contradiction that the number of output channels is larger than that of input channels in the units. To fix it, a simple operator called *channel split* was introduced, which equally split input channels into two branches.  Nevertheless, while doing spatial downsampling, *channel split* is removed in order to double the number of output channels.

Also, the three successive element-wise operations, *concat*, *channel shuffle* and *channel split*, can be merged into a single element-wise operation. 

<img src="https://tva1.sinaimg.cn/large/00831rSTgy1gcladgdc1pj316q0pudlj.jpg" alt="image-20200229173424082" style="zoom:67%;" />

![image-20200229173955686](https://tva1.sinaimg.cn/large/00831rSTgy1gclba0fekrj315i0n2q9i.jpg)




## MobileNetV3
Time flies to June 2019. The most cutting-edge and GPU-consuming technology, **network architecture search (NAS)** has been being applied widely, and gradually reducing job positions in the DL market. This generation of MobileNets [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244) is tuned to mobile phone CPUs. 

![image-20200304155730379](https://tva1.sinaimg.cn/large/00831rSTgy1gcladh8d5cj30gl0b276u.jpg)



### Block-wise search

First, For block-wise search in **Large** mobile model, we employ a hardware-aware NAS approach which is similar to **MnasNet** and get similar results. Therefore, we simply reuse the same **MnasNet-A1** as our initial Large model.

Then, for **Small** mobile model, we change multi-objective reward as follow:

<img src="https://www.zhihu.com/equation?tex=ACC(m) \times {[LAT(T) / TAR]}^w" alt="ACC(m) \times {[LAT(T) / TAR]}^w" class="ee_img tr_noresize" eeimg="1">

where 
<img src="https://www.zhihu.com/equation?tex=ACC(m)" alt="ACC(m)" class="ee_img tr_noresize" eeimg="1">
 is accuracy, 
<img src="https://www.zhihu.com/equation?tex=LAT(m)" alt="LAT(m)" class="ee_img tr_noresize" eeimg="1">
 is model's latency, and 
<img src="https://www.zhihu.com/equation?tex=TAR" alt="TAR" class="ee_img tr_noresize" eeimg="1">
 stands for target latency. Specially, 
<img src="https://www.zhihu.com/equation?tex=w=-0.15" alt="w=-0.15" class="ee_img tr_noresize" eeimg="1">
 (vs 
<img src="https://www.zhihu.com/equation?tex=w=-0.07" alt="w=-0.07" class="ee_img tr_noresize" eeimg="1">
 in MnasNet) to  compensate for the larger accuracy change for different latencies.  Upon this, we apply NetAdapt and other optimizations to obtain the final **MobileNetV3-Small** model.



### Layer-wise search

For layer-wise search, we modify this **NetAdapt** algorithm in MnasNet and minimize the ratio between latency change and accuracy change 
<img src="https://www.zhihu.com/equation?tex=(\frac{\Delta Acc}{\Delta latency})" alt="(\frac{\Delta Acc}{\Delta latency})" class="ee_img tr_noresize" eeimg="1">
.  In short, the technique proceeds as follows:

> 1. Starts with a seed network architecture found by platform-aware NAS.
>
> 2. For each step:
>
>   (a) Generate a set of new *proposals*. Each proposal represents a modification of an architecture that generates at least  
<img src="https://www.zhihu.com/equation?tex=\delta=0.01|L|" alt="\delta=0.01|L|" class="ee_img tr_noresize" eeimg="1">
(
<img src="https://www.zhihu.com/equation?tex=L" alt="L" class="ee_img tr_noresize" eeimg="1">
 is the latency of the seed model) reduction in latency compared to the previous step.
>
>   (b) For each proposal we use the pre-trained model from the previous step and populate the new proposed architecture, truncating and randomly initializing missing weights as appropriate. Finetune each proposal for 
<img src="https://www.zhihu.com/equation?tex=T=1000" alt="T=1000" class="ee_img tr_noresize" eeimg="1">
 steps to get a coarse estimate of the accuracy.
>
>   (c) Selected best proposal according to some metric.
>
> 3. Iterate previous steps until target latency is reached.



### Redesigning Expensive Layers and Nonlinearities

![image-20200304143534725](https://tva1.sinaimg.cn/large/00831rSTgy1gcladhoo2fj30f508cwfs.jpg)

The original last stage uses *inverted residual* to expand features, followed by *average pooling* and 
<img src="https://www.zhihu.com/equation?tex=1 \times 1" alt="1 \times 1" class="ee_img tr_noresize" eeimg="1">
 convolution (dense layer), to get the final prediction.  Now, we move *average pooling* forward, and the final set of features is now computed at 
<img src="https://www.zhihu.com/equation?tex=1 \times 1" alt="1 \times 1" class="ee_img tr_noresize" eeimg="1">
 spatial resolution instead of 
<img src="https://www.zhihu.com/equation?tex=7 \times 7" alt="7 \times 7" class="ee_img tr_noresize" eeimg="1">
 spatial resolution. This approach reduces 7ms latency and almost no hurt in accuracy.

Besides, we replace the filter number of the first 
<img src="https://www.zhihu.com/equation?tex=3 \times 3" alt="3 \times 3" class="ee_img tr_noresize" eeimg="1">
 convolution layer (to produce initial features from image) from 32 to 16.

***Swish*** is an effective substitute for *ReLU*, which can improve accuracy in many neural networks. However, the sigmoid part in *swish* is quite expensive to compute.

<img src="https://www.zhihu.com/equation?tex=swish(x) = x*\sigma(x)" alt="swish(x) = x*\sigma(x)" class="ee_img tr_noresize" eeimg="1">

We replace the original sigmoid function to hard-sigmoid: 
<img src="https://www.zhihu.com/equation?tex=\frac{ReLU6(x+3)}{6}" alt="\frac{ReLU6(x+3)}{6}" class="ee_img tr_noresize" eeimg="1">
, then we have:

<img src="https://www.zhihu.com/equation?tex=h\mbox{-}swish(x) = x*\frac{ReLU6(x+3)}{6}" alt="h\mbox{-}swish(x) = x*\frac{ReLU6(x+3)}{6}" class="ee_img tr_noresize" eeimg="1">

![image-20200304153544169](https://tva1.sinaimg.cn/large/00831rSTgy1gcladeum78j30er057gmb.jpg)



### Large squeeze-and-excite

 We set all the size of the **squeeze-and-excite** bottleneck to fixed to be 
<img src="https://www.zhihu.com/equation?tex=1/4" alt="1/4" class="ee_img tr_noresize" eeimg="1">
 of the number of channels in expansion layer, without discernible latency cost. We should also note that the *FCs* can be replaced by convolution layers.

![image-20200304154452538](https://tva1.sinaimg.cn/large/00831rSTgy1gcladkgiupj30ek07h0ty.jpg)



The final MobileNetV3-Large and MobileNetV3-Small model:

![image-20200304154512410](https://tva1.sinaimg.cn/large/00831rSTgy1gcladibb8xj30el0g3q66.jpg)

![image-20200304154520838](https://tva1.sinaimg.cn/large/00831rSTgy1gclade36pvj30ef0bs76d.jpg)