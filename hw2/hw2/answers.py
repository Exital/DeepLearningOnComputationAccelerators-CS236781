r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd = 0.12
    lr = 0.1
    reg = 0.09
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.09, 0.001, 0.004, 0.0002, 0.001

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr_vanilla=lr_vanilla, lr_momentum=lr_momentum,
                lr_rmsprop=lr_rmsprop, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0.11, 0.000095
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

(1) Dropout is a regularization technique with the goal of preventing overfitting.
When the model overfits, we expect to see small train loss while the validation loss is higher.
Comparing the dropout results against the non-dropout results shows that the training loss increases by a small margin, while the test loss decreases. This implies the model is prone to overfitting.

(2) The graph shows that low dropout settings has a higher test loss than what high dropout settings produces.
Both have pretty much the same accuracy, which means that the model with the higher dropout is more stable.
We can Infer from the above that them model is prone to overfitting (probably due to larger quantity of parameters and lack of depth).
Even when we used dropout=0.8, a big gap between the train and test accuracy is still apparent.
The model is apparently not suitable for the CIFAR10 dataset.
"""

part2_q2 = r"""
**Your answer:**
Yes, it is possible.
Our model predicts a class for an image by taking the maximum over the class score and the loss is calculated by the formula: $-log(\frac{e^{x[class]}}{\sum_j e^{x[j]}})$
If the probability of the wrong classified examples increases, yet the overall correct predictions stays the same, the accuracy will stay the same as well, and the loss will increase.

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
(1)
The deeper the depth of the network, the higher our chances to capture more significant features of the images in the training set.
As the depth increases we suffer from multiple phenomenon’s such as vanishing gradient and overfitting (higher number of parameters).
Deeper depths requires bigger datasets and more epochs to reach it’s potential.
We can see for example, that with L = 16, it takes about 60 epochs to reach the same result L = 8 accomplished with 30 epochs.
For conclusion, increasing the depth will result with a better accuracy, up to a certain point.

(2)
For L = 2, 4, 8 we can see the model has reached a plateau and invoked an early stopping. We assume not enough features were captured for the model in order to make an optimal learning due to a lower depth.
According to the graph, it seems the lower L’s are prone to overfitting as well.
One suggestion to overcome it is use dynamic learning rate. Instead of using early stopping, when the same conditions are met, we would reduce the learning rate.
However, we would have to use early stopping if the learning rate is getting to small, to prevent numerical errors on one hand, and a small learning rate has almost no effect on the process anyway.
A second suggestion, using various regularization methods such as dropout.

Also, an obvious solution would be to increase the size of the dataset (with different images), or applying transformations on the given images of the original dataset.
"""

part3_q2 = r"""
**Your answer:**
Analysis:
•	Per L, the greater the number of filters, the better the accuracy is.
•	Per L, greater K also reduce the amount of epochs required to reach the best accuracy.
•	At L = 8, K = 256 we see the model has been compromised. We assume it has gone through overfitting since the train loss is extremely low compare to the others, while the test loss is high compare to the others.
Compare:
In experiment 1.1 we saw a more distinct difference of the processes per L. In this experiment, per L, each process of a different K had a subtle change from it’s previous.
We can see on this experiment, when L = 8, K = 256 that when the model has overfitting, the test loss is very high, and yet the accuracy continues to increase, which confirms the reasoning we introduced for question 2 on the second notebook.


"""

part3_q3 = r"""
**Your answer:**
Analysis:
We see the results are more stable since each network contains various depths for the conv filters.
Also, all of the models have come to an early stopping around 30 epochs.
It seems that the more complicated the network is (number of filters) the worst it performs on this particular dataset.

"""

part3_q4 = r"""
**Your answer:**
Analysis:
The residual networks are more reliable, we see that the train approximately matches the test loss, and the same thing happens with the accuracy.
As post analysis note, we probably should have used a lower learning rate here since no model have come to an early stopping and could have continued learning furthermore.

Compare to 1.1:
In 1.1 we see no drastic changes when increasing L, however, in this experiment, higher L shows a big difference with the learning rate.
With lesser L we need much less epochs than higher L.
Although it was apparent in 1.1, here we can see this phenomenon clearly.

Compare to 1.3:
Similarly, to 1.3 we see that the more complex the model is (number of filters wise) it performs poorly. In this experiment, with lesser filters the models could have used more epochs in order to improve, but when included larger number of filters, they came to an early stopping before descending to worst accuracies. It is visible too on the train loss, we can see more room to improve with lesser filters.



"""

part3_q5 = r"""
**Your answer:**
(1)
We created a costume convolution block. We took 3 convolution of dimensions 1x1, 3x3 and 5x5, and extracted the mean of them, following with batch normalization.
We’ve added residual blocks with 2D dropout and added dropout layers to the classifier.
Additionally, we have replaced the ReLU function with leaky ReLU.
Finally, we’ve replaced the max pooling with dilated convolutions except for the first layer.


(2)
Analysis:
Generally, we can say the model stable. Ignoring small deviations, the graphs of the test and train loss appear to be similar, and same could be said for the test and train accuracies.


Compare to experiment 1:
Our model had room to improve, whereas in experiment 1, for the most part, we came across several occasions of early stoppings.
In our model, the differences between number of filters is noticeable, and in experiment 1, sometimes the margin was low, to the point it didn’t matter (1.1, 1.2, 1.3).
In our model, there is not immediate hint of overfitting, or underfitting, as opposed to (1.2).
The trajectory of the accuracy and loss is stable, there is no sharp changing, the models is heading towards a goal target.
When we look at experiment 1, we sometime see and erratic behavior, graphs crossing each other, and changing directions. This is another points which led us conclude out model is generally stable.



"""
# ==============
