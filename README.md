gon-pytorch
===========

Unofficial PyTorch implementation of Gradient Origin Networks (Bond-Taylor & Willcocks, 2021).

![](./figs/gon.png)

| Reconstructions | Samples |
| ----------------| ------- |
| ![](./figs/fashionmnist-recons.png) | ![](./figs/fashionmnist-samples.png) |
| ![](./figs/mnist-recons.png) | ![](./figs/mnist-samples.png) |


Usage
-----

Requirements:
- PyTorch
- Hydra

```bash
python train_gon.py dataset.name=<MNIST|FashionMNIST> dataset.root=<data-root>
```
