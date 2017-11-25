import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    predicted_labels = model.predict(np.c_[xx.ravel(), yy.ravel()].T)
    predicted_labels = predicted_labels.reshape(xx.shape)

    plt.contourf(xx, yy, predicted_labels, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=np.squeeze(y), cmap=plt.cm.Spectral)
    plt.show()

def compute_accuracy(model, X, true_Y):
    m = X.shape[1]
    predicted_labels = model.predict(X)
    return np.sum((predicted_labels == true_Y) / m)

def compute_accuracy_multilabel(model, X, true_Y):
    m = X.shape[1]
    predicted_labels = model.predict(X)
    argmax_preds = np.argmax(predicted_labels, axis=0)
    argmax_y = np.argmax(true_Y, axis=0)
    return np.sum((argmax_preds == argmax_y) / m)
