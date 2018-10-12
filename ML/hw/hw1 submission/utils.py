import cv2
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

def convert2pixel_value(filename):
    """
    Converts the filename into a flatten numpy array of the raw pixel values.
    Input:
    - filename (string): name of the file to be processed.
    """
    raw_img = cv2.imread(filename)

    return raw_img.flatten()

def convert2color_mean_std(filename):
    """
    Converts the filename into the mean and standard deviation of the different color channels (RGB).
    Input:
    - filename (string): name of the file to be processed.
    """
    raw_img = cv2.imread(filename)

    return cv2.meanStdDev(raw_img)

def convert2color_hist(filename, bin):
    """
    Converts the filename into a histogram with bin bins for each of the different color channels (RGB). The concatenated
    vector of the different color histogram is returned.
    Input:
    - filename (string): name of the file to be processed.
    - bin: number of bins for the histogram of each color channel.
    """
    raw_img = cv2.imread(filename)
    hist = []
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr = cv2.calcHist([raw_img],[channel],None,[bin],[0,256])
        hist.append(histr)
    return np.concatenate(hist)

def convert2color_3Dhist(filename, bin = 8):
    """
    Converts the filename into a 3D histogram. The 3D histogram values are then flattened to return a single dimension array.
    Input:
    - filename (string): name of the file to be processed.
    - bin: number of bins for the histogram of each color channel.
    """
    raw_img = cv2.imread(filename)
    histr = cv2.calcHist([raw_img],[0,1,2],None,[bin]*3,[0,256]*3)

    return histr.flatten()

def load_data(path, feature = 'raw'):
    """
    Loads data into pixel values from the list of path given. Returns
    Input:
    - path (list): list of path to load the data from.
    - feature: either 'raw' or 'hist' for raw pixel values and 3D histogram respectively.
    """
    x_train=[]
    y_train=[]
    for c,i in enumerate(path):
        os.chdir(i)
        l = os.listdir()
        for i in l:
            if feature == 'raw':
                vf = convert2pixel_value(i)
            else:
                vf = convert2color_3Dhist(i)
            x_train.append(vf)
            y_train.append(c)

    x_train = np.concatenate([i[np.newaxis] for i in x_train])
    y_train = np.array(y_train)

    # comment below to remove the shuffling of the data
    arr = np.arange(x_train.shape[0])
    np.random.shuffle(arr)
    x_train = x_train[arr]
    y_train = y_train[arr]

    return x_train, y_train

def sigmoid(x):
    """
    Applies the sigmoid function on the given vector.
    Input(s):
    - x : numpy vector of values
    """
    return 1/(1+np.exp(-x))

def add_bias(dataset):
    """
    Add a one to each sample for bias. Dataset must be of the form rows: samples, columns: features
    """
    n, m = dataset.shape
    out = np.ones((n, m+1))
    out[:,:-1] = dataset
    return out

def initialize_params(size=3073, seed=123):
    """
    Initialize parameters W weights and b biases.
    Input(s):
    - size (int): size of the parameters
    - seed (int): seed for the random number generator
    """
    rng = np.random.RandomState(seed)

    return rng.normal(size=(size,))

def log_loss(x_train, y_train, W):
    """
    Computes the loss value of the logistic loss.
    Input(s):
    - x_train, y_train: training data and labels. x_train takes different forms depending on the features used and y_train
                        is {-1,+1}.
    - W: value of the parameters.
    """
    z = y_train * np.dot(x_train, W)
    h = sigmoid(z)

    return -np.mean(np.log(h))

def log_grad(x_train, y_train, W):
    """
    Computes the gradient of the logistic loss function.
    Input(s):
    - x_train, y_train: training data and labels. x_train takes different forms depending on the features used and y_train
                        is {-1,+1}.
    - W: value of the parameters.
    """
    z = y_train * np.dot(x_train, W)
    h = sigmoid(z)
    n = x_train.shape[0]

    return 1/n * np.dot(x_train.T,(y_train * (h-1)))

def next_batch(x_train, y_train, batch_size=2):
    """
    Returns a batch of size batch_size for stochastic gradient descent.
    - x_train, y_train: training data and labels. x_train takes different forms depending on the features used and y_train
                        is {-1,+1}.
    - batch_size (int): size of each batch.
    """
    for i in np.arange(0, x_train.shape[0], batch_size):
        yield (x_train[i:i+batch_size],y_train[i:i+batch_size])

def log_classifier(x, learnt_W):
    """
    Takes in the test set and learnt parameters and returns the accuracy of the classifier on the test set.
    Inputs:
    -
    """
    return (sigmoid(np.dot(x, learnt_W)) >= .5) * 2 - 1

def log_accuracy(x, y, learnt_W):
    """
    Returns the accuracy of the model with parameters learnt_W.
    Input(s):
    -
    """
    output = log_classifier(x, learnt_W)
    return np.sum(np.absolute(y - output) == 0)/y.shape[0]

def log_train(x_train, y_train, x_test, y_test, W, alpha=0.01, batch_size = 4, epoch = 100):
    """
    Trains the model with given learning rate alpha, batch_size, epoch and initialized parameters W.
    """
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    W_history = []
    for e in np.arange(epoch):
        epoch_loss = []
        for (batchx, batchy) in next_batch(x_train, y_train):
            loss = log_loss(batchx, batchy, W)
            grad = log_grad(batchx, batchy, W)
            epoch_loss.append(loss)
            W += -alpha * grad
        loss_history.append(np.average(epoch_loss))
        train_acc = log_accuracy(x_train, y_train, W)
        test_acc = log_accuracy(x_test, y_test, W)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        W_temp = W.copy()
        W_history.append(W_temp)

    return loss_history, train_acc_history, test_acc_history, W_history    

def display_plot(loss_history, train_acc_history, test_acc_history):
    """
    Plots out the loss function value for the different epochs and the accuracy of the test set for the
    parameters learnt in the different epochs.
    Input(s):
    - loss_history: list containing the change in the loss value over the epochs
    - acc_history: list containing the accuracy value of the test set over the epochs
    """
    f, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
    ax1.plot(loss_history, color='r')
    # ax1.plot(model_log.history['val_acc'])
    ax1.set_title('Loss (Lower Better)')
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.legend(['train', 'validation'], loc='upper right')

    ax2.plot(train_acc_history)
    ax2.plot(test_acc_history)
    ax2.set_title('Accuracy (Higher Better)')
    ax2.set(xlabel='Epoch', ylabel='Accuracy')
    ax2.legend(['train', 'test'], loc='center right')
    
def para_diff(learnt_W_history):
    """
    Plots the change in the value of the parameter learnt and returns the epoch with the smallest change.
    Input(s):
    - learnt_W_history: history of the different value of the parameter at different epoch.
    """
    diff = []
    for i in range(0,len(learnt_W_history)-1):
        d = np.linalg.norm(learnt_W_history[i]-learnt_W_history[i+1])/np.linalg.norm(learnt_W_history[i])
        diff.append(d)

    plt.plot(diff)
    plt.title('Change in parameter')
    plt.xlabel('Epoch')
    plt.ylabel('Difference')
    print ('Epoch with minimum change in parameter: {} '.format(np.argmin(diff)))
    
def hinge_loss(x_train, y_train, theta):
    """
    Returns the hinge loss function values
    Input(s):
    - x_train, y_train: train data and labels
    - theta: parameters for the model
    """
    z = y_train * np.dot(x_train, theta)
    return np.average(np.maximum(1-z,0))

def hinge_grad(x_train, y_train, theta):
    """
    Returns the gradient of the hinge loss function at theta
    Input(s):
    - x_train, y_train: train data and labels
    - theta: parameters for the model
    """
    z = y_train * np.dot(x_train, theta)
    grad = -(y_train * x_train.T).T
    grad[z>1]=0
    return np.average(grad, axis=0)

def hinge_classifier(x, learnt_theta):
    """
    Performs classification using hinge loss
    Input(s):
    - x: data for classification
    - learnt_theta: learnt parameters 
    """
    return np.sign(np.dot(x,learnt_theta))

def hinge_accuracy(x, y, learnt_theta):
    """
    Returns accuracy of the hinge loss
    Input(s):
    - x: data for classification
    - y: label of the data
    - learnt_W: learnt parameters 
    """
    output = hinge_classifier(x, learnt_theta)
    return np.sum(np.absolute(y - output) == 0)/y.shape[0]   

def hinge_train(x_train, y_train, x_test, y_test, theta, alpha=0.01, batch_size = 1, epoch = 100):
    """
    Trains the hinge loss model with given learning rate alpha, batch_size, epoch and initialized parameters W.
    Returns the history of the loss, train accuracy, test accuracy and the value of the parameters 
    at different epochs.
    Input(s):
    - x_train, y_train: train data and labels
    - x_test, y_test: test data and labels
    - W: parameters of the model
    - alpha: learning rate
    - batch_size: batch size for stochastic gradient descent
    - epoch: number of times the dataset is passed through the model.
    """
    loss_history = []
    train_acc_history = []
    test_acc_history = []
    theta_history =[]
    for e in np.arange(epoch):
        epoch_loss = []
        for (batchx, batchy) in next_batch(x_train, y_train):
            loss = hinge_loss(batchx, batchy, theta)
            grad = hinge_grad(batchx, batchy, theta)
            epoch_loss.append(loss)
            theta += -alpha * grad
        loss_history.append(np.average(epoch_loss))
        train_acc = log_accuracy(x_train, y_train, theta)
        test_acc = log_accuracy(x_test, y_test, theta)
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        theta_temp = theta.copy()
        theta_history.append(theta_temp)
    
    return loss_history, train_acc_history, test_acc_history, theta_history

def relabel(y, c):
    """
    Relabel labels y from multiple classes to binary classes.
    Input(s):
    - y: labels to be relabeled
    - c: the class to be labeled as 1
    """
    new_label = np.ones(y.shape)
    ind = y != c
    new_label[ind] = -1
    return new_label

def relabel_multiclass(y):
    """
    Uses relabel to relabel the labels of data with multiple classes to multiple binary labels. Returns a 
    list of relabeled labels, with each item in the list a binary label (-1/+1) for each class.
    Input(s):
    - y: labels to be relabeled
    """
    y_list = []
    c = len(np.unique(y))
    for i in np.arange(c):
        y_temp = y.copy()
        y_temp = relabel(y_temp, c=i)
        y_list.append(y_temp)
    
    return y_list

def onevsall_train(x_train, y_train, x_test, y_test, W, alpha=0.01, batch_size = 4, epoch = 100):
    """
    Trains the parameters of each one-vs-all model. Returns a list of learnt_W_history, with each item of the list 
    belonging to a certain model. Each item in the list contains the learnt_W_history that spans over the chosen number
    of epochs for a certain model.
    Input(s):
    - x_train: training images
    - y_train: labels for the training images
    - x_test: testing images
    - y_test: labels for the testing images
    - W: parameters of the model
    - alpha: learning rate
    - batch_size: size of each batch using stochastic gradient descent
    - epoch: number of times the whole dataset is used to train the model
    """

    learnt_W_history_list = []
    y_train_list = relabel_multiclass(y_train)
    y_test_list = relabel_multiclass(y_test)
    
    for i in np.arange(len(y_train_list)):
        W_temp = W.copy()
        loss_history, train_acc_history, test_acc_history, learnt_W_history = log_train(x_train, y_train_list[i], 
                                                                                        x_test, y_test_list[i], 
                                                                                        W_temp, 
                                                                                        epoch=epoch, 
                                                                                        batch_size=batch_size)
        learnt_W_history_list.append(learnt_W_history)
    
    return learnt_W_history_list

def onevsall_predict(x, learnt_W_history_list):
    """
    Input(s):
    - x: data to be predicted
    - learnt_W_history_list: history of learnt parameters at different epoch
    """
    predict_epoch = []
    for i in range(len(learnt_W_history_list[0])): # loop over epoch

        prob_list = [] # stores list of prob for each model for a given epoch
        for j in range(len(learnt_W_history_list)): # loop over models
            p = sigmoid(np.dot(x,learnt_W_history_list[j][i])) # get probabilites for the models at epoch i
            prob_list.append(p)

        prob = np.concatenate([i[np.newaxis] for i in prob_list])  
        predict = np.argmax(prob, axis = 0) # predicts the class for epoch i
        predict_epoch.append(predict) # stores the prediction from the model at epoch i at predict_epoch
    return predict_epoch

def onevsall_accuracy(y, predict_epoch):
    """
    Input(s):
    - y: true label of the data
    """
    acc_list = []
    for i in range(len(predict_epoch)):
        acc = np.average(predict_epoch[i] == y)
        acc_list.append(acc)
    return acc_list        