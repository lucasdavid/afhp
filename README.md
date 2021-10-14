# Painting by Numbers

Adversarial Generation of Fine-Art Paintings.


## Building and Running

Add the data to the `./data` folder (or create a symbolic link with `ln -s /path/to/dataset data`).

The project can be build and run with the following commands:
```shell
$ docker build -t ldavid/painting-by-numbers .
$ docker run -it --rm -v $(pwd)/notebooks:/tf/notebooks -v $(pwd)/data:/data -p 8888:8888 ldavid/painting-by-numbers
```
