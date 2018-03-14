## Using Deep Learning to Predict Water Point functionality from an Image

A [Jupyter notebook](https://nbviewer.jupyter.org/github/pld/dl4d/blob/master/image-classification.ipynb) with the Keras model that we used to predict water point functionality from an image. The images we used are not publicly available but you can run the same experiments against the [Kaggle Cats vs. Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data). The image preparation code that we've used is available in this [fork of gigasquid's kagge-cats-dogs](https://github.com/pld/kaggle-cats-dogs) repository.



Default model is a feed forward neural network with four 2D convolutional layers of 32, 32, 64, and 128 nodes, all using a rectified linear unit activation function and a 2 by 2 pool size. This is followed by a fully connected layer of 128 nodes and a dropout layer to prevent overfitting. Finally, we use a sigmoid activation function to model the single class binary output of functioning or not-functioning.

<img alt="model architecture" src="https://blog.ona.io/assets/images/2018-02-28/split_handpump_Adam_le-4_do05_32_100.png" width="250"/>

On our dataset the above model obtains 77.7% validation accuracy after 100 epochs:

<img alt="model performance" src="https://blog.ona.io/assets/images/2018-02-28/split_handpump_Adam_le-4_do05_32_100_performance.png" width="500px"/>

Samples images and neural network activations:

<img alt="images and neural network activations" src="https://blog.ona.io/assets/images/2018-02-28/images_activations.png" width="500px"/>

### Future Work

There are also a number of improvements to the model that we can experiment with. The current hyper-parameters were chosen with a coarse grid search, we should do a fine grid search over the hyper-parameters and add additional hyper-parameters to the search space, including pool size and optimizer. We should experiment with additional network architectures, both deeper and wider. Although preliminary tests did not produce good results, we should also experiment further with transfer learning using pre-existing network weights, such as ImageNet.

### Further Reading and References

1. Duchi, Hazan, and Singer, [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)
2. Guss and Salakhutdinov, [On Characterizing the Capacity of Neural Networks using Algebraic Topology](https://arxiv.org/abs/1802.04443)
3. Kingma and Ba, [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980v8)
