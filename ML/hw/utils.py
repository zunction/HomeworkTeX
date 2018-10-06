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
# def log_train(x_train, y_train, x_test, y_test, W, alpha=0.01, batch_size = 4, epoch = 100):
#     """
#     Trains the model with given learning rate alpha, batch_size, epoch and initialized parameters W.
#     """
#     loss_history = []
#     train_acc_history = []
#     test_acc_history = []
#     for e in np.arange(epoch):
#         epoch_loss = []
#         for (batchx, batchy) in next_batch(x_train, y_train):
#             loss = log_loss(batchx, batchy, W)
#             grad = log_grad(batchx, batchy, W)
#             epoch_loss.append(loss)
#             W += -alpha * grad
#         loss_history.append(np.average(epoch_loss))
#         train_acc = log_accuracy(x_train, y_train, W)
#         test_acc = log_accuracy(x_test, y_test, W)
#         train_acc_history.append(train_acc)
#         test_acc_history.append(test_acc)
#
#     return loss_history, train_acc_history, test_acc_history, W

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
