# The Numerics of GANs
This repository contains the code to reproduce the core results from the paper [The Numerics of GANs](https://arxiv.org/abs/1705.10461).

To cite this work, please use
```
@INPROCEEDINGS{Mescheder2017NIPS,
  author = {Lars Mescheder and Sebastian Nowozin and Andreas Geiger},
  title = {The Numerics of GANs},
  booktitle = {Advances in neural information processing systems},
  year = {2017}
}
```

# Dependencies
This project uses Python 3.5.2. Before running the code, you have to install
* [Tensorflow 1.0](https://www.tensorflow.org/)
* [Numpy](http://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [Matplotlib](http://matplotlib.org/)
* [tqdm](https://pypi.python.org/pypi/tqdm)

These dependencies can be installed using pip by running
```
pip install tensorflow-gpu numpy scipy matplotlib tqdm
```

# Instructions
First download [cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and put them into the `./data` directory (in seperate folders). The `experiments` folder contains scripts for starting the different experiments.