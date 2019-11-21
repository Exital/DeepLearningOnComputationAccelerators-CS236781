r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
No. Consider the following scenario.
Let's say a test sample drop inside a small cluster of train samples with the correct label.
However, for a very large K, we would have to evaluate the labels of clusters at a distance from out test sample.
and if K is greater than sum of smaples in our small correct cluster and enough samples on the distanced clusters, we would have
to label our samples with a wrong class according to the majority of K samples, which is being highjacked by the distance clusters.

On the other way, very small K might be susceptible to noise. That is why $k = 3$ is better than $k = 1$.

"""

part2_q2 = r"""
**Your answer:**
1.  If we do train each model on all of the dataset, then we would get an 100% accuracy for $k = 1$.
    Every "test" samples has l2_dist == 0 from himself (it is part of the memoriezed trainnig set).
    This will give us an extremely overfitted model.

2.  Selecting the best model with respect to a test-set will give us a model biased by the given test-set.
    If we would try to test the model on an unseen dataset, we might get unreliable results because the hyperparametr K was
    preempted chosen in order to preform well on the first test-set.


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The $\Delta > 0$ hamming distance is selected in an arbitrary way in attempt to avoid over-fitting to the trainning set.
"""

part3_q2 = r"""
**Your answer:**
The model is learning flattened images. 
The data, as far as the model concerns, is serialized and it detects sequnces of data and make it's "assumptions" on it.
That is why the model is sensitive to translations and rotations of the images.

If we look at some of the errors:

* We could easily mistake the 5 with the closed circle at the bottom of it to a 6, and so does the model.
* Another error which fits to the model's way of learning is the mistake between 2 and 6. It could be easily be interperted as each other if we transpose the image.
* the 7 image in the last-1 line is closer to a 9 beacuse of the additional line which drops from it's top.

The interpertaion of this model is similar to KNN in the following way:
KNN classifies each object according to clusters in it's memoriezd samples data.
If for example, we have a cluster of many 6's and a few 2's, then the KNN would classify the 2's which might drop
close to that cluster as a 6.

However, KNN is less sensitive to noise, as opposed to SVM.
"""

part3_q3 = r"""
**Your answer:**
1.  Good.
    *   With a slow learning rate, the model will not converge at all. We would see both graphs with a distance between them.
    *   With a high learning rate, the graph will have a spiked oscillator.


2.  Slightly underfitted.
    We can see a slight acsent around the 30th epoch which suggests that the model can still improve.


"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The pattern we expect to see is a a narrow scatter within the sleeve of the avarge.
After the feature engineering our model is more centralized around the avarge in a more compound way within the sleeve.
This suggests it is more accurate, due to the feature engineering.

The more the plot is concentrated inside the sleeve, the fitter the model is.

Also, we would want to see the test samples closer to the center of the sleeve rather than the trainning samples.

"""

part4_q2 = r"""
**Your answer:**
1.  With logspace we get to check values in different order of magnitued, rather then values in the same order.
2.  20 times with $\lambda$, 3 times per degree, and 3 times per folds:  $total = 3 \cdot 3 \cdot 20 = 180$

"""

# ==============
