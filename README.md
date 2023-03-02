# ConvSearch: Pseudo-Inverted Bottleneck Convolution for DARTS Search Space

**This work has been accepted to the 2023 IEEE International Conference on Acoustic, Speech and Signal Processing for oral presentation**

Checkout our arXiv preprint [Paper](https://arxiv.org/abs/2301.01286)
> A. Ahmadian, Y. Fei, L. S. P. Liu, K. N. Plataniotis, and M. S. Hosseini, ‘Pseudo-Inverted Bottleneck Convolution for DARTS Search Space’. arXiv, 2023.


## Abstract

Differentiable Architecture Search (DARTS) has attracted considerable attention as a gradient-based Neural Architecture
Search (NAS) method. Since the introduction of DARTS,
there has been little work done on adapting the action space
based on state-of-art architecture design principles for CNNs.
In this work, we aim to address this gap by incrementally augmenting the DARTS search space with micro-design changes
inspired by ConvNeXt and studying the trade-off between accuracy, evaluation layer count, and computational cost. To this
end, we introduce the Pseudo-Inverted Bottleneck conv block
intending to reduce the computational footprint of the inverted
bottleneck block proposed in ConvNeXt. Our proposed architecture is much less sensitive to evaluation layer count and
outperforms a DARTS network with similar size significantly,
at layer counts as small as 2. Furthermore, with less layers,
not only does it achieve higher accuracy with lower GMACs
and parameter count, GradCAM comparisons show that our
network is able to better detect distinctive features of target
objects compared to DARTS.

## Highlights
**[C1.]** We present an ***incremental experiment procedure*** depicted in Fig.1 to
evaluate how design components from `ConvNeXt` impact the
performance of `DARTS` by redesigning its search space.

**[C2.]** We introduce a ***Pseudo-Inverted Bottleneck block*** shown in Fig 2 (c) to
implement an inverted bottleneck structure while minimizing
model footprint and computations. This outperforms vanilla
`DARTSV2` with *lower number of layers*, *parameter count*, and
*GMACs*.

<div align="center">
<a align="center">
  <img alt="roadmap" src="Figures/methodology_v2.png" width="50%" height="40%"></img>
  <br>
  <em>Fig. 1: Roadmap of the incremental augmentations along with their corresponding accuracies and
methodologies.</em>
</a>
<br><br>
<a align="center">
<img alt="conv1" src="Figures/convnextblk_norm.png" width="20%" height="20%" hspace="20"></img>
<img alt="conv2" src="Figures/darts_sepconvblk_norm.png" width="20%" height="20%" hspace="10"></img>
<img alt="conv3" src="Figures/newconvblk_norm.png" width="20%" height="20%" hspace="10"></img>
<br>
<em> Fig. 2: Convolution Blocks : (a) DARTS Separable Convolution Block; (b) Inverted Bottleneck ConvNeXt Convolution
Block (Cinv = C × 4); (c) Pseudo-Inverted Bottleneck Cell
(Cinv = C × 2)</em>
</a>
</div>


## Environment Setup
- Required Computer Vision datasets: CIFAR-10
  - We used CIFAR-10 for both architecture searching and training from scratch
- Requird Modules in `requirements.txt`

## Methodology
- The following adaptations from ConvNeXt and Swin Transformer are made:
  - Replace ReLU with GeLU: *boosts accuracy by 0.12%*
  - Replacing BatchNorm with LayerNorm: *accuracy degradation*
  - Adapting the ConvNeXt Block: 
    1) Reducing num. of activation and normalization layers
    2) Adapting to an inverted bottleneck structure (from [MobileNetV2](https://arxiv.org/abs/1801.04381))
    3) Moving up the depthwise separable conv. layer to facilitate training with large kernel size.

