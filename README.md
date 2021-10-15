# Painting by Numbers

Adversarial Generation of Fine-Art Paintings.


## Building and Running

Add the data to the `./data` folder (or create a symbolic link with `ln -s /path/to/dataset data`).

The project can be build and run a notebook server instance with the following commands:
```shell
$ docker build -t ldavid/painting-by-numbers .
$
$ docker run -it --rm -p 8888:8888 \
    -v $(pwd)/code:/tf/code \
    -v $(pwd)/data/cached/.keras:/root/.keras \
    -v $(pwd)/notebooks:/tf/notebooks \
    -v $(pwd)/data:/tf/data \
    -v $(pwd)/logs:/tf/logs \
    -v $(pwd)/config:/tf/config \
    ldavid/painting-by-numbers
```

To run experiments, simply use the `-d` option in `docker run -itd ...` and
call the scripts `docker exec -it {INSTANCE_ID} python {PATH_TO_SCRIPT}`.

For example:
```shell
$ docker run -itd --rm -p 8888:8888 -v $(pwd)/code:/tf/code -v $(pwd)/notebooks:/tf/notebooks -v $(pwd)/data:/tf/data -v $(pwd)/data/cached/.keras:/root/.keras -v $(pwd)/logs:/tf/logs -v $(pwd)/config:/tf/config ldavid/painting-by-numbers
6290cd3658b4779df48f5586f0e8587a63838319e638d6ad88587654b537a0bd

$ docker exec -it 6290 python /tf/code/adversarial_feature_hallucination_cifar100.py with /tf/config/docker/cifar100/adv_feature_hallucination_fsl.baseline.yml -F /tf/logs/cifar100/afh_baseline
```


## Implemented

 #  | Dataset            | Strategy                                   | Description                    | Ref.
--- | ------------------ | ------------------------------------------ | ------------------------------ | --------------------------------------------------
1   | Cifar100           | Adversarial Feature Hallucination FSL      |                                | [2003.13193](https://arxiv.org/pdf/2003.13193.pdf)
2   | Painter by Numbers | Supervised Contrastive Learning (baseline) | Multi-task between artist, style and genre classification, producing a better embedding space | [2004.11362](https://arxiv.org/abs/2004.11362)
