# Binary Image Classifier

## *Jack Koltermann, Gabriel Lahman, and Michael Stavinsky*


#### Dependencies

* Pillow PIL Fork [Pillow — Pillow (PIL Fork) 5.3.0 documentation](https://pillow.readthedocs.io/en/5.3.x/)
* NumPy [NumPy](https://www.google.com/search?client=safari&rls=en&q=numpy&ie=UTF-8&oe=UTF-8)
* Keras [Keras Documentation](https://keras.io) (Only used for comparison)
* TensorFlow [TensorFlow Documentation](https://www.tensorflow.org) (Only used for comparison)

#### Components

* DatasetAugmenter
* ImageAugmenter
* classifier
* data
* layer
* kerasHotdog and hotdogNetwork.h5
* network
* structure
* utilities


#### DatasetAugmenter

The DatasetAugmenter class is used to automatically process, augment, resize, and save images used for training. This
is achieved through the use of the ImageAugmenter class. After giving a directory of images and the target size for the 
images within the directory, several types of augmentations such as gaussian blur or rotation are applied to each image. 
The edited images are renamed and then saved in the directory specificed by the user.



#### ImageAugmenter

Utilized by the DatasetAugmenter class to perform augmentation on individual images. The augmentations include: scaling,
padding, rotation, and filtering. When scaling, the aspect ratio is preserved in order to minimize distortion of the images. 
Filters that can be applied are: Gaussian blur, smoothing, sharpening, and normal blur.



#### Classifier

This is the upper level class of the image classifier as a whole. A classifier object allows you to define a network, train
said network, and classify any image. When defining the network, the user must provide the following: a file defining the 
structure of the netowrk, a folder to save network weights in, the text conversion of numbers pertaining to image
classes, the data to be fed through the network, and hyperparameters used in training.



#### Data

To provide a helpful wrapper class for the data used in the network, the Data class provides several advantages. First, as 
opposed to loading the entire training dataset into memory before training, it instead loads and provides batches as needed 
during training. These batches are determined by the setBatches function, which partitons the data according the batch size.
The setBatches function would be called during every epoch of training in order to reset and reshuffle the batches. However, 
all data can be returned using the getAll function, helpful when running the test data during training.



#### Layer

Within layer is contained all of the different layer types within our convolutional network. 

*Conv*

The Conv object represents a convolutional layer. This object stores information of specific layers such as the filters (weights),
filter size, padding, and output size. Using this stored information, the Conv object also provides the `feed` functionality,
which accepts an input matrix, performs a convolution operation on this matrix, and storing the result. In addition to 
storing the result of the convolution, the input matrix is also stored within the class for use during backprpogation.

*Relu*

The Relu object represents a Rectified Linear Unit within the network. Specifically, our implementation is known as a 
leaky RELU (lRELU). lRELU attempts to alleviate the issue of having too sparse of data by using an small, constant
hyperparameter on negative values to prevent them from being just 0. Similar to the Conv layer, the Relu object also
stores its last output, but also stores the locations of values in the input that were greater than zero, which is used
to calculate the gradient during backpropgation.

*FC*

The FC object represents a fully-connected layer. This layer also stores similar information to the Conv layer, but contains
a vectorized activation function, for which we used the sigmoid function. Additionally, dropout within the fully connected
layers is implemented, which randomly sets some outputs of the layer to 0 according to a specificed dropout percentage, in
order to reduce overfitting.

*Pool*

The Pool object represents a pooling layer, specifically a max pooling. Max pooling reduces the width and heigth size of the 
data by only passing forward the max value within a given receptive field. The depth/number of channels are not affected.
The location of the values passed forward are stored within the object for use during backpropogation.



#### kerasHotdog

Model implemented in keras using same architecture as the one built by us. Trained model is saved in hotdogNetwork.h5.



#### Network

This object defines the actual convolutional neural network representation. It is full breadth and encompasses all necessary 
functionality to load the network structure, construct the network using the objects from the layer file, train the network,
save the weights after training, and load in pre-trained weights. The code is best explained by reading the interal docstrings
and comments. Guiding principles for the backpropogation are explained [here](https://pdfs.semanticscholar.org/5d79/11c93ddcb34cac088d99bd0cae9124e5dcd1.pdf).



#### Structure

A CSV file defining the structure of the network and read utilized by the network class. Requirements are defined within the
structureGuide.txt file



#### Utilities

Contains miscellaneous functions for relu, weight initialization, and calculating output size, with explanations contained 
within code.
