import abc
import torch


class ClassifierLoss(abc.ABC):
    """
    Represents a loss function of a classifier.
    """

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    @abc.abstractmethod
    def loss(self, *args, **kw):
        pass

    @abc.abstractmethod
    def grad(self):
        """
        :return: Gradient of the last calculated loss w.r.t. model
            parameters, as a Tensor of shape (D, C).
        """
        pass


class SVMHingeLoss(ClassifierLoss):
    def __init__(self, delta=1.0):
        self.delta = delta
        self.grad_ctx = {}

    def loss(self, x, y, x_scores, y_predicted):
        """
        Calculates the Hinge-loss for a batch of samples.

        :param x: Batch of samples in a Tensor of shape (N, D).
        :param y: Ground-truth labels for these samples: (N,)
        :param x_scores: The predicted class score for each sample: (N, C).
        :param y_predicted: The predicted class label for each sample: (N,).
        :return: The classification loss as a Tensor of shape (1,).
        """

        assert x_scores.shape[0] == y.shape[0]
        assert y.dim() == 1

        # TODO: Implement SVM loss calculation based on the hinge-loss formula.
        #  Notes:
        #  - Use only basic pytorch tensor operations, no external code.
        #  - Full credit will be given only for a fully vectorized
        #    implementation (zero explicit loops).
        #    Hint: Create a matrix M where M[i,j] is the margin-loss
        #    for sample i and class j (i.e. s_j - s_{y_i} + delta).
        # ====== YOUR CODE: ======
        
        margins = x_scores -torch.unsqueeze(x_scores[range(0,x_scores.shape[0]),y],dim=1)
        idx = margins==0
        margins = margins +self.delta
        margins[idx] = 0
        margins[margins<0] = 0
        temp = margins.sum(dim=1)
        
        # ========================
        # TODO: Save what you need for gradient calculation in self.grad_ctx
        # ====== YOUR CODE: ======
        binary = margins
        binary[margins>0] = 1
        self.grad_ctx['binary'] = binary
        self.grad_ctx['data'] = x.t()
        self.grad_ctx['GT'] = y
        # ========================
        loss = torch.mean(temp)
        return loss

    def grad(self):
        """
        Calculates the gradient of the Hinge-loss w.r.t. parameters.
        :return: The gradient, of shape (D, C).

        """
        # TODO:
        #  Implement SVM loss gradient calculation
        #  Same notes as above. Hint: Use the matrix M from above, based on
        #  it create a matrix G such that X^T * G is the gradient.

        # ====== YOUR CODE: ======
        y = self.grad_ctx['GT']
        binary = self.grad_ctx['binary']
        row_sum = binary.sum(dim=1)
        binary[range(0,binary.shape[0]),y] = -row_sum.t()
        G = binary
        x =   self.grad_ctx['data']
        grad = x@G
        grad = grad/len(y)
        # ========================

        return grad
