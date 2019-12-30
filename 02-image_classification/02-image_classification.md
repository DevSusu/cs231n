# Lec02: Image Classification

### Assignment 1
* k-nearest neighbor
* svm, softmax
* two-layer neural networks
* image feature

### Image Classification
Given a Image and a pre-defined set of labels, you determine the correct label

* Semantic Gap:
* Viewpoint variation
* Illumination
* Deformation
* Occlusion: only part of the object is seen
* Background Clutter
* Intraclass variation

Thus this is a **Challenging** task to do.


### Implementation
```python3
def classify(image):
    # do something
    return label
```
some **fixed** algorithm has been attempted, but wasn't so effective

thus data data-driven approach
```python3
def train(images, labels):
    # ML
    return model

def predict(model, test_imgs):
    # ...
    return test_labels
```

### K-Nearest Neighbors
Memorize all data & label, find most similar image and return

* DataSet: [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
* Very Simple algorithm
* Comparing distance of Images: L1 distance
* We want fast for prediction, thus OK for slow training
* Limitation of nearest neighbor
  * K-Nearest Neighbors. majority vote from K neighbor
* Distance Metric: L1(diamond) & L2(circle)
  * [Demo](http://vision.stanford.edu/teaching/cs231n-demos/knn/)

The important thing is to change **K** and **Distance Metric**.
How do you choose this *hyperparameters*? -> Just try all out and find which value works best.

#### Setting Hyperparameters
* Select what works best on given data -> terrible idea. training always results in K=1
* Split train & test data. -> terrible again. Machine Learning should be working on **new data**
* Split train & validation & test data. -> Better. Pick learning model that are well working on validation data set. Then go through test data.
* Cross-Validation. Validation data is randomly selected. Used on small datasets. NEVER used in practice.
* Once selected, hyper-parameters are only computed again if extra 1% performance is important

#### KNN is never used
* slow at test time
* distance metric is poorly representing similarity
* curse of dimensionality. larger the dimension is, exponential amount of data is needed


### Linear Classification
* Building blocks of Neural Network
* Example of image captioning. 1 neural network looking at image, 1 neural network knowing language

* Back to CIFAR10
* Parametric Approach
don't keep the hole data, use only parameters on predicting time
```
image : 32*32*3 vector x
parameters of weights W (10 * 1027)

f(x,W) -> 10 numbers giving scores.
```

* Simple Linear Classifier : f(x,W) = Wx + b
* W can give a clue on what's going on.
  * Linear Classifier sums up all categories into one template.
  * Linear Classifying is drawing a line in high dimensional space.
* Non-linear cases (square, circle, ..) are hard cases
  * odd even pair cases
  * multiple models
* How do you select W properly?? -> Loss function, Optimization, CNN
