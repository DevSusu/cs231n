# Lec03: Loss Functions & Optimization

### Review of Last class
* Challenges of image classification
* data-driven approach
* kNN
* Linear Classifier
  * column vector
  * parameter matrix W
  * template per class
  * line in high dimensional space
* TODO
  * define a **loss function** that quantifies our unhappiness
  * come up with a efficient procedure to find the loss function(**optimize**)

### Multiclass SVM Loss function
* loss function tells how bad is our current classifier
```
 L = 1/N * sum_i^{ L_i(f(x_i,W),y_i) }

 f : prediction function
 y_i : real label(correct answer)
 L_i : loss function
```
* if correct (score_yi >= score_j + 1) then loss is 0
* else then score_j - score_yi + 1
* graph looks like a hinge `\_`

* cat, car, frog SVM loss example
* Q. how do you select the **+1**? in fact the absolute value really doesn't matter. Change is important
* min loss is 0. max is INF
* with small initial W, if all s~=0 then loss is (# of category)-1
* if loss functions were summing on correct class, ... not important
  * loss being 0 is just elegant
* what if we used mean instead of sum. -> no difference.
* what if we computes squared sums? -> very different. but used in some cases
* if L = 0 at some W, is W unique? no it's not unique. (ex 2W)
  * but training data performance is not so important.
  * we should avoid overfitting, so Loss being 0 isn't so important
  * adding **Regularization term** model should be simple.

#### Regularization
* L2, L1 regularization
* elastic net (L1+L2)
* Max norm, Dropout, Batch normalization, stochastic depth ...
* giving space to your models and prevents overfitting
* L2 reg example
  * w1 = [1,0,0,0] and w2 = [0.25,0.25,0.25,0.25]
  * L2 spreads the effect(sparse). selects w2
  * L1 has the opposite effect.

### Softmax Classifier (Multinomial Logistic Regression)
* for SVM, we spit out scores but only cared about if its maximum
* in softmax, scores = unnormalized log probabilities of the classes
* take log on a Conditional probability function
* minimize the negative log likelihood
* maximizing the log is mathematically easier than the raw value
* minimum loss if 0( -log(0) ), maximum loss if INF
* at first iteration(s ~= 0), loss is log(c).
* Difference
  * svm gives up after reaching some point
  * softmax loss continues to push and get better

#### Recap
* dataset x,y
* score function f(x;W) = Wx
* loss func: softmax, svm + R(W)



### Optimization
* we talked about evaluating the score of W
* how do we find W?
* guy walking around the valley to find a way down
* there's really not much hope on finding a analytic method
* 1. Random Search
* 2. Follow the slope
  * feel the ground's local slope
  * actually works out well, and used a lot
  * recall a derivative function. in multivariable: `gradient`
  * dot product with direction and gradient. steepest is the `negative gradient`

#### Gradient
* W , W+h , dW(gradient) explanation
* instead of computing with every variable, use classic calculus(analytic gradient)
  * dW = some function derived from loss function
  * use the numerical gradient as debugging
* Gradient Descent
  * eval gradient, W -= step_size * gradient
  * step size is the most important hyperparameter
  * there are other fancy algorithms to update W
* Stochastic Gradient Descent(SGD)
  * for big N such as imagenet, computing sum of loss can be heavy
  * thus in practice, for each iteration, we sample some small set of training examples
  * from the mini batch, we estimate the total sum
* Gradient Descent Interactive demo

#### Image Features
* putting raw pixel values into linear
* idea of two-step data feeding was brought up. from pixels, we drive features out and pass it to linear classifiers
* feature transform such as coordinate transform...
* Color Histogram
* Histogram of Oriented Gradients
  * compute dominant edges inside each 8*8 pixels
  * similar to what human vision uses
* Bag of Words
  * intuition from NLP. understanding paragraph from words.
  * extract random patches from image and cluster
    * this includes histogram + direction
* Image Classification Pipeline
  * image -> feature extraction -> linear classifier
  * very similar on ConvNet, it's just that ConvNet learns the features itself and computing weights on the hole layer, not just the linear classifier
