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
**[C1.]** We present an ***incremental experiment procedure*** to
evaluate how design components from `ConvNeXt` impact the
performance of `DARTS` by redesigning its search space.

**[C2.]** We introduce a ***Pseudo-Inverted Bottleneck block*** to
implement an inverted bottleneck structure while minimizing
model footprint and computations. This outperforms vanilla
`DARTSV2` with *lower number of layers*, *parameter count*, and
*GMACs*.

## Datasets
- CV datasets: CIFAR-10

## Methodology
This is the first phase. Seaching code: `cnn/train_search_rmsgd.py`
Need to specify:
- Dataset to search on
- Training options like optimizer, learning rates, batch size, etc
- Model architecture details like init channel size, # layers, # nodes, etc.


Example (To search for our final genotype -- `NEWCONV_design_cin4_cifar10_DARTSsettings` on `CIFAR-10`):
```
cd ConvSearch/cnn
python .py \
--dataset ADP-Release1 --image_size 64 \
--adas --scheduler_beta 0.98 \
--learning_rate 0.175 --batch_size 32 \
--layers 4 --node 4 \
--unrolled \
--file_name adas_ADP-Release1_size_64_lr_0.175_beta_0.98_layer_4_node_4_unrolled
```

When searching is finished, you need to copy/paste the generated *genotype* into `cnn/genotypes.py` and name it, in order to continue the following step.

## Evaluation
This is the second phase. Train the searched architecture from scratch. Code: `cnn/train_cifar.py`, `cnn/train_cpath.py`. Need to speify:
- Dataset to train on
  - In `cnn/train_cifar.py`, pass argument `--cifar100` to train on CIFAR100; otherwise CIFAR10 is used.
  - In `cnn/train_cpath.py`, pass argument `--dataset $DATASET` where `$DATASET` can be either ADP, BCSS, BACH, or OS.
- Model architecture to train. Pass argument `--arch $MODEL` where `$MODEL` is the *genotype* name stored in `cnn/genotypes.py`.
- Other model details including # layers, # init channels.
- Training options including batch size, learning rate, etc.

Example (to train DARTS_ADP_N4 on ADP):
```
cd path_to_this_repo/cnn
python train_cpath.py \
--dataset ADP --image_size 272 \
--arch DARTS_ADP_N4 --layers 4 \
--batch_size 96 --epochs 600 \
--auxiliary --cutout 
```
