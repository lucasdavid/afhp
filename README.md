# Adversarial Feature Hallucination in a Supervised Contrastive Space for Few-Shot Learning of Provenance in Paintings

Official implementation of the paper "Adversarial Feature Hallucination in a Supervised Contrastive Space for Few-Shot Learning of Provenance in Paintings", presented in LA-CCI 2023.

![Proposal Overview](assets/afhn_overview.png)
    
## Summary

Adversarial Feature Hallucination Networks for Few-Shot Learning ([AFHN](https://arxiv.org/abs/2003.13193))
over a [Supervised Contrastive](https://proceedings.neurips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf) space,
applied to Few-Shot tasks sampled from the [Painter by Numbers](https://www.kaggle.com/c/painter-by-numbers) dataset.

## Building and Running

Add the data to the `./data` folder (or create a symbolic link with `ln -s /path/to/dataset data`).

You can run all configurations with:
```shell
$ ./runners/1-run-all.sh
```

To run the best configuration (our approach, last configuration in [run-all.sh](./runners/1-run-all.sh)):
```shell
$ python run.py --data_split frequent --strategy supcon_mh --backbone_train_epochs 0
```

## Citation

If our code be useful for you, please consider citing our paper using the following BibTeX entry.

```
@INPROCEEDINGS{10409405,
  author={David, Lucas and Liborio, Luis and Paiva, Guilherme and Severo, Marianna and Valle, Eduardo and Pedrini, Helio and Dias, Zanoni},
  booktitle={2023 IEEE Latin American Conference on Computational Intelligence (LA-CCI)},
  title={Adversarial Feature Hallucination in a Supervised Contrastive Space for Few-Shot Learning of Provenance in Paintings}, 
  year={2023},
  volume={},
  number={},
  pages={1-6},
  doi={10.1109/LA-CCI58595.2023.10409405}}
```
