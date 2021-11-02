# Adversarial Feature Hallucination in a Supervised Contrastive Space for Few-Shot Learning of Provenance in Paintings

Official implementation of the paper "Adversarial Feature Hallucination in a Supervised Contrastive Space for Few-Shot Learning of Provenance in Paintings", presented in LA-CCI 2023.

![Proposal Overview](assets/afhn_overview.png)
    
## Summary

Adversarial Feature Hallucination Networks for Few-Shot Learning ([AFHN](https://arxiv.org/abs/2003.13193))
over a [Supervised Contrastive](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) space,
applied to Few-Shot tasks sampled from the [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) dataset.

## Building and Running

Add the data to the `./data` folder (or create a symbolic link with `ln -s /path/to/dataset data`).

You can run the project with:
```shell
$ ./runners/1-train.sh
```
