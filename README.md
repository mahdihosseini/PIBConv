# Adaptive DARTS

## Datasets
In `data` folder:
- CV datasets: CIFAR-10, CIFAR-100
- CPATH datasets: ADP, BCSS, BACH, OS

## Searching
This is the first phase. Seaching code: `cnn/train_search_adas.py`
Need to specify:
- Dataset to search on
- Training options like optimizer, learning rates, batch size, etc
- Model architecture details like init channel size, # layers, # nodes, etc.

Example (to search for DARTS_ADP_N4 on ADP):
```
cd path_to_this_repo/cnn
python train_search_adas.py \
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
