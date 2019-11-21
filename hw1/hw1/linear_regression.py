import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, 'weights_')
        # TODO: Calculate the model prediction, y_pred
        # ====== YOUR CODE: ======
        y_pred = X@self.weights_
        # ========================
        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution
        #  Use only numpy functions. Don't forget regularization.
        
        # ====== YOUR CODE: ======
        N = X.shape[0]
        reg_mat = N*self.reg_lambda*np.eye(X.shape[1])
        reg_mat[0,0] = 0
        inv_mat = np.linalg.inv( X.transpose()@X + reg_mat)
        w_opt = inv_mat@X.transpose()@y
        # ========================
        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)
        
        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().
        
        
        # ====== YOUR CODE: ======
        xb = np.hstack((np.ones((X.shape[0],1)),X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2,reg_lambda=0.1):
        self.degree = degree
        
        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        self.reg_lambda = reg_lambda
        # ========================

    def fit(self, X, y=None):
        N = len(y)
        reg_mat = N*self.reg_lambda*np.eye(X.shape[1])
        reg_mat[0,0] = 0
        #reg_mat[:,:]=0
        inv_mat = np.linalg.inv(X.transpose()@X + reg_mat)
        w_opt = inv_mat@(X.transpose()@y)
        self.weights_ = w_opt
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').
        
        # ====== YOUR CODE: ======
        poly = PolynomialFeatures(self.degree)
        data = X
        X_transformed = poly.fit_transform(data)
        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======
    
    df_corr =  df.corr().drop_duplicates().unstack().abs().sort_values(ascending=False)     
    df_corr_topk = df_corr[target_feature].iloc[1:n+1].reset_index()
    df_corr_topk = DataFrame(df_corr_topk)
    top_n_features = df_corr_topk['index'].values
    top_n_corr = df_corr_topk[0].values
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """
    
    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    e = y-y_pred
    err = (1/(len(y)))*(np.linalg.norm(e)**2)
    mse = err
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    e = y - y_pred
    mse = np.linalg.norm(e)
    mse = mse*mse    
    
    mean_y = y.mean()
    err_y = y-mean_y
    err_y_score = np.linalg.norm(err_y)*np.linalg.norm(err_y)
    
    r2 = 1-mse/err_y_score
    # ========================
    return r2


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    #reg_lambda_key = 'bostonfeaturestransformer__reg_lambda'
    #degree_key = 'bostonfeaturestransformer__degree'
    kf = sklearn.model_selection.KFold(n_splits = k_folds,shuffle = True)
    
    best_lambda = 0
    best_degree = 1
    lambda_list = []
    degree_list = []
    #scores = []
    for train_index,test_index in kf.split(X):
        best_err = np.Inf
        for lamda in lambda_range:
            for degree in degree_range:
                curr_params = {'bostonfeaturestransformer__reg_lambda': lamda,
                               'bostonfeaturestransformer__degree':degree,
                               'linearregressor__reg_lambda': lamda
                               }
                model.set_params(**curr_params)
                curr_model = model.fit(X[train_index],y[train_index])
                curr_score = mse_score(y[test_index],curr_model.predict(X[test_index]))
                if curr_score<best_err:

                    best_err = curr_score
                    best_lambda = lamda
                    best_degree = degree
         

        lambda_list.append(best_lambda)
        degree_list.append(best_degree)
        
    

    lambda_list = np.array(lambda_list)
    degree_list = np.array(degree_list)            
    best_params = {'bostonfeaturestransformer__reg_lambda': lambda_list.mean(),
                   'bostonfeaturestransformer__degree':int(round(degree_list.mean())),
                   'linearregressor__reg_lambda': lambda_list.mean()
    }   

    # ========================

    return best_params
