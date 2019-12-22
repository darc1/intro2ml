#################################
# Your name: Chen Dar
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

"""
Q4.1 skeleton.

Please use the provided functions signature for the SVM implementation.
Feel free to add functions and other code, and submit this file with the name svm.py
"""

# generate points in 2D
# return training_data, training_labels, validation_data, validation_labels
def get_points():
    X, y = make_blobs(n_samples=120, centers=2, random_state=0, cluster_std=0.88)
    return X[:80], y[:80], X[80:], y[80:]


def create_plot(X, y, clf):
    plt.clf()

    # plot the data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.PiYG)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0] - 2, xlim[1] + 2, 30)
    yy = np.linspace(ylim[0] - 2, ylim[1] + 2, 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])


def train_three_kernels(X_train, y_train, X_val, y_val):
    """
    Returns: np.ndarray of shape (3,2) :
                A two dimensional array of size 3 that contains the number of support vectors for each class(2) in the three kernels.
    """
    linear_clf = svm.SVC(kernel='linear', C=1000)
    linear_clf.fit(X_train, y_train)
    lin_support = linear_clf.n_support_
    create_plot(X_train, y_train, linear_clf)
    plt.title(f"Linear Number of Support Vectors={sum(lin_support)}")
    plt.savefig("section-a-linear.png")
    plt.show()
    quadratic_clf = svm.SVC(kernel='poly', degree=2, C=1000)
    quadratic_clf.fit(X_train, y_train)
    quad_support = quadratic_clf.n_support_
    create_plot(X_train, y_train, quadratic_clf)
    plt.title(f"Quadratic Number of Support Vectors={sum(quad_support)}")
    plt.savefig("section-a-quadratic.png")
    plt.show()
    rbf_clf = svm.SVC(kernel='rbf', C=1000)
    rbf_clf.fit(X_train, y_train)
    rbf_support = rbf_clf.n_support_
    create_plot(X_train, y_train, rbf_clf)
    plt.title(f"RBF Number of Support Vectors={sum(rbf_support)}")
    plt.savefig("section-a-rbf.png")
    plt.show()
    return np.array([lin_support, quad_support, rbf_support])


def linear_accuracy_per_C(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    c_values = [10**i for i in range(-5,6)]
    results_validation = []
    results_training = []
    i=-5
    for c in c_values:
        clf = svm.SVC(kernel='linear', C=c).fit(X_train, y_train)
        results_validation.append(clf.score(X_val, y_val))
        results_training.append(clf.score(X_train, y_train))
        # create_plot(X_val, y_val, clf)
        # plt.title(f'Linear\n C=10^{i}')
        # plt.savefig(f"C-accuracy-10-{i}.png")
        # plt.show()
        i+=1;
    # plt.plot(c_values, results_training, marker='x', color='r', label='Training Accuracy')
    # plt.plot(c_values, results_validation, marker='o', color='c', label='Validation Accuracy')
    # plt.ylabel('Validation Accuracy')
    # plt.xlabel('C')
    # plt.title('Accuracy as a function of C')
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig(f"C-accuracy.png")
    # plt.show()

    return np.array(results_validation)



def rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val):
    """
        Returns: np.ndarray of shape (11,) :
                    An array that contains the accuracy of the resulting model on the VALIDATION set.
    """
    gamma_values = [10**i for i in range(-5,6)]
    results_validation = []
    results_training = []
    i=-5
    for gamma in gamma_values:
        clf = svm.SVC(kernel='rbf', C=10, gamma=gamma).fit(X_train, y_train)
        results_validation.append(clf.score(X_val, y_val))
        results_training.append(clf.score(X_train, y_train))
        # create_plot(X_val, y_val, clf)
        # plt.title(f'RBF\n GAMMA=10^{i}')
        # plt.savefig(f"GAMMA-accuracy-10-{i}.png")
        # plt.show()
        i+=1
    # plt.plot(gamma_values, results_training, marker='x', color='r', label='Training Accuracy')
    # plt.plot(gamma_values, results_validation, marker='o', color='c', label='Validation Accuracy')
    # plt.ylabel('Validation Accuracy')
    # plt.xlabel('Gamma')
    # plt.title('Accuracy as a function of GAMMA')
    # plt.xscale('log')
    # plt.legend()
    # plt.savefig(f"GAMMA-accuracy.png")
    # plt.show()

    return np.array(results_validation)

# if __name__ == "__main__":
#     X_train, y_train, X_val, y_val = get_points()
#     train_three_kernels(X_train, y_train, X_val, y_val)
#     linear_accuracy_per_C(X_train, y_train, X_val, y_val)
#     rbf_accuracy_per_gamma(X_train, y_train, X_val, y_val)