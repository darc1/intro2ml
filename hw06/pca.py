import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np


def plot_vector_as_image(image, h, w, title):
    """
    utility function to plot a vector as image.
    Args:
    image - vector of pixels
    h, w - dimesnions of original pi
    """
    plt.imshow(image.reshape((h, w)), cmap=plt.cm.gray)
    plt.title(title, size=12)
    plt.show()


def get_pictures_by_name(name='Ariel Sharon'):
    """
    Given a name returns all the pictures of the person with this specific name.
    YOU CAN CHANGE THIS FUNCTION!
    THIS IS JUST AN EXAMPLE, FEEL FREE TO CHANGE IT!
    """
    lfw_people = load_data()
    selected_images = []
    n_samples, h, w = lfw_people.images.shape
    target_label = list(lfw_people.target_names).index(name)
    for image, target in zip(lfw_people.images, lfw_people.target):
        if (target == target_label):
            image_vector = image.reshape((h * w, 1))
            selected_images.append(image_vector)
    return selected_images, h, w


def load_data():
    # Don't change the resize factor!!!
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    return lfw_people


######################################################################################
"""
Other then the PCA function below the rest of the functions are yours to change.
"""


def PCA(X, k):
    """
    Compute PCA on the given matrix.

    Args:
        X - Matrix of dimesions (n,d). Where n is the number of sample points and d is the dimension of each sample.
        For example, if we have 10 pictures and each picture is a vector of 100 pixels then the dimesion of the matrix would be (10,100).
        k - number of eigenvectors to return

    Returns:
      U - Matrix with dimension (k,d). The matrix should be composed out of k eigenvectors corresponding to the largest k eigenvectors
            of the covariance matrix.
      S - k largest eigenvalues of the covariance matrix. vector of dimension (k, 1)
    """
    cov = np.matmul(np.matrix.transpose(X), X)
    w, v = np.linalg.eig(cov)
    k_largest = np.argsort(w)[::-1][:k]
    v = np.matrix.transpose(v)
    U = v[k_largest]
    S = w[k_largest]
    return U, S


# def get_X():
#     selected_images, h, w = get_pictures_by_name('Ariel Sharon')
#     X = np.array(selected_images)[:, :, 0]
#     center_data(X)
#     return X, h, w
#
# def center_data(X):
#     mean = np.mean(X, axis=0)
#     for i in range(len(X)):
#         X[i] = X[i] - mean
#
#
# def plot_vectors(U, h, w, rows, cols, fig_name):
#     plt.figure(figsize=(1.5 * cols, 2.2 * rows))
#     plt.subplots_adjust(0.6, 0.5, 1.5, 1.5)
#     for k in range(U.shape[0]):
#         plt.subplot(rows, cols, k + 1)
#         plt.imshow(U[k].reshape((h, w)), cmap=plt.cm.gray)
#         plt.yticks(())
#         plt.xticks(())
#     plt.tight_layout()
#     plt.savefig(f"{fig_name}.png")
#     plt.show()
#
#
# def get_all_pics():
#     lfw_people = load_data()
#     selected_images = []
#     labels = []
#     n_samples, h, w = lfw_people.images.shape
#     for image, target in zip(lfw_people.images, lfw_people.target):
#         image_vector = image.reshape((h * w, 1))
#         selected_images.append(image_vector)
#         labels.append(list(lfw_people.target_names)[target])
#     return selected_images, labels, h, w

# def b(X, h ,w):
#     U, S = PCA(X, 10)
#     plot_vectors(U, h, w, 2, 5, "section-b")


# def c(X, h, w):
#     dists = []
#     k_vals = [1, 5, 10, 30, 50, 100]
#     for k in k_vals:
#
#         U, S = PCA(X, k)
#         V = np.matmul(X, np.matrix.transpose(U))
#         vX = np.matmul(np.transpose(U), np.transpose(V))
#         vX = np.matrix.transpose(vX)
#         random_pics = np.random.choice(len(X), 5)
#
#         plot_vectors(np.concatenate((X[random_pics], vX[random_pics])), h, w, 2, 5, f"section-c-k-{k}")
#         dists.append(np.sum([np.linalg.norm(X[i] - vX[i]) for i in random_pics]))
#
#     plt.plot(k_vals, dists, color='r', marker='o')
#     plt.xlabel('K')
#     plt.ylabel('L2 Distances')
#     plt.title('L2 Distances by K')
#     plt.savefig('section-c.png')
#     plt.show()


# def d():
#     all_data, labels, h, w = get_all_pics()
#     X = np.array(all_data)[:, :, 0]
#     center_data(X)
#     X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=0)
#     k_vals = [1, 5, 10, 30, 50, 100, 150, 300]
#     scores = []
#     for k in k_vals:
#         U, S = PCA(X_train, k)
#         A = np.matmul(X_train, np.matrix.transpose(U))
#         vX_test = np.matrix.transpose(np.matmul(U, np.matrix.transpose(X_test)))
#         clf = svm.SVC(kernel='rbf', C=1000, gamma=10 ** -7).fit(A, y_train)
#         scores.append(clf.score(vX_test, y_test))
#     plt.plot(k_vals, scores, color='g', marker='x')
#     plt.xlabel('K')
#     plt.ylabel('Accuracy')
#     plt.title('Accuracy by K')
#     plt.savefig('section-d.png')
#     plt.show()

# if __name__ == "__main__":
    # X, h, w = get_X()
    # b(X, h, w)
    # c(X, h, w)
    # d()
