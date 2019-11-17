from sklearn.datasets import fetch_openml
import numpy.random
import numpy
import json

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


def knn(images, image_labels, query_image, k):
    distances = numpy.zeros(len(images))

    for i in range(len(images)):
        # calculate Euclidean distance
        dist = numpy.linalg.norm(images[i] - query_image)
        distances[i] = dist

    # find k nearest distances index labels
    k_nearest_idx = numpy.argpartition(distances, k)
    k_nearest_labels = image_labels[k_nearest_idx[:k]]

    # count labels occurrences from k nearest and return label with max
    (values, counts) = numpy.unique(k_nearest_labels, return_counts=True)
    nearest_label_idx = numpy.argmax(counts)
    return values[nearest_label_idx]


def run_for_k(k, n):
    print(f"running kNN for {n} images with k={k}")
    correct_predictions = 0
    for i in range(n):
        prediction = knn(train, train_labels, test[i], k=k)
        if prediction == test_labels[i]:
            correct_predictions += 1

    print(f"number of correct predictions: {correct_predictions}")
    correct_predictions_rate = float(correct_predictions) / float(n)
    print(f"correct predication percentage for k:{k} is: {correct_predictions_rate}")
    return correct_predictions_rate


def return_results(results, file_name=None):
    if file_name is None:
        return results

    with open(file_name, "w+") as w:
        w.write(json.dumps(results))


def section_b(file_name=None):

    print("Running section (b)")
    result = {"prediction_rate" : run_for_k(10, 1000)}
    return return_results(results=result, file_name=file_name)


def section_c(file_name=None):
    print("Running section (c)")

    results = {}
    for k in range(1, 101):
        results[k] = run_for_k(k=k, n=1000)

    print(f"{results}")
    return return_results(results=results, file_name=file_name)


def section_d(file_name=None):
    print("Running section (d)")

    results = {}
    for n in range(100, 5100, step=100):
        results[n] = run_for_k(k=1, n=n)

    print(f"{results}")
    return return_results(results=results, file_name=file_name)


section_b(file_name="sectionB.json")
section_c(file_name="sectionC.json")
section_d(file_name="sectionD.json")