import numpy


def knn(train_images, labels, testimage, k):
    distances = numpy.zeros(len(train_images))

    for i in range(len(train_images)):
        # calculate Euclidean distance
        dist = numpy.linalg.norm(train_images[i] - testimage)
        distances[i] = dist

    # find k nearest distances index labels
    k_nearest_idx = numpy.argpartition(distances, k)
    k_nearest_labels = labels[k_nearest_idx[:k]]

    # count labels occurrences from k nearest and return label with max
    (values, counts) = numpy.unique(k_nearest_labels, return_counts=True)
    nearest_label_idx = numpy.argmax(counts)
    return values[nearest_label_idx]
