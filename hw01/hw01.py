import json

import numpy
from joblib import Parallel, delayed
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

from nn import knn

mnist = fetch_openml('mnist_784')
data = mnist['data']
labels = mnist['target']

idx = numpy.random.RandomState(0).choice(70000, 11000)
train = data[idx[:10000], :].astype(int)
train_labels = labels[idx[:10000]]
test = data[idx[10000:], :].astype(int)
test_labels = labels[idx[10000:]]


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


def run_job(key, k, n):
    return [key, run_for_k(k=k, n=n)]


def return_results(results, file_name=None):
    if file_name is None:
        return results

    with open(file_name, "w+") as w:
        w.write(json.dumps(results))


def create_graph(results, x_label, y_label, file_name):
    x_values = [v[0] for v in results]
    y_values = [(1 - v[1]) for v in results]
    plt.plot(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(file_name)


def section_b(file_name=None):
    print("Running section (b)")
    result = {"prediction_rate": run_for_k(10, 1000)}
    return return_results(results=result, file_name=file_name)


def section_c(file_name=None):
    print("Running section (c)")

    results = Parallel(n_jobs=16)(delayed(run_job)(k, k, 1000) for k in range(1, 3))
    # results = Parallel(n_jobs=16)(delayed(run_job)(k, k, 1000) for k in range(1, 101))
    print(results)

    create_graph(results, "n", "error rate", "section-c.png")

    return return_results(results=results, file_name=file_name)


def section_d(file_name=None):
    print("Running section (d)")

    results = Parallel(n_jobs=16)(delayed(run_job)(n, 1, n) for n in range(100, 5100, 100))
    print(results)
    create_graph(results, "k", "error rate", "section-d.png")

    return return_results(results=results, file_name=file_name)


# section_b(file_name="sectionB.json")
section_c(file_name="sectionC.json")
section_d(file_name="sectionD.json")
