from skeleton_perceptron import *
from skeleton_sgd import *


def section_1_a():
    n_values = [5, 10, 50, 100, 500, 1000, 5000]
    num_runs = 100
    accuracy = {}
    permutations = []

    for i in range(num_runs):
        permutations.append(np.random.permutation(len(train_data)))

    for n in n_values:
        print(f"running for n={n}")
        results = np.ndarray(shape=(num_runs,), dtype=float)
        for i in range(num_runs):
            permutation = permutations[i][:n]
            w = perceptron(data=train_data[permutation], labels=train_labels[permutation])
            results[i] = check_w_accuracy_perceptron(w, test_data, test_labels)

        accuracy[n] = results

    for n, v in accuracy.items():
        print(f"n={n}, accuracy: mean={np.mean(v)}, percentiles 5th={np.percentile(v, 5)} 95th={np.percentile(v, 95)}")


def section_1_b():
    w = perceptron(train_data, train_labels)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.title("w as image")
    plt.savefig("section-1-b.png")
    plt.show()
    plt.clf()

    success_images = run_check_perceptron(w, test_data, test_labels)[2]

    positive_images = list(filter(lambda v: v[0] == 1, success_images))
    fig = plt.figure(figsize=(28, 28))
    row, col = 4, 4
    plt.title(f"Classified: positive Label: 1")
    for i in range(0, min(len(positive_images), row * col)):
        image_1 = positive_images[i]
        fig.add_subplot(row, col, i + 1)
        plt.imshow(np.reshape(test_data[image_1[2]], (28, 28)), interpolation='nearest')
    plt.savefig("section-1-b-classified-positive-successfully.png")
    plt.show()
    plt.clf()

    negative_images = list(filter(lambda v: v[0] == -1, success_images))
    fig = plt.figure(figsize=(28, 28))
    row, col = 4, 4
    plt.title(f"Classified: negative Label: -1")
    for i in range(0, min(len(negative_images), row * col)):
        image_1 = negative_images[i]
        fig.add_subplot(row, col, i + 1)
        plt.imshow(np.reshape(test_data[image_1[2]], (28, 28)), interpolation='nearest')
    plt.savefig("section-1-b-classified-negative-successfully.png")
    plt.show()
    plt.clf()


def section_1_c():
    w = perceptron(train_data, train_labels)
    accuracy = check_w_accuracy_perceptron(w, test_data, test_labels)
    print(f"full train data w accuracy over test data: {accuracy}")


def section_1_d():
    w = perceptron(train_data, train_labels)
    failure_images = run_check_perceptron(w, data=test_data, labels=test_labels)[1]

    image_1 = failure_images[1]
    plt.imshow(np.reshape(test_data[image_1[2]], (28, 28)), interpolation='nearest')
    plt.title(f"Classified: {image_1[1]} Label: {image_1[0]}")
    plt.savefig("section-1-d-img1.png")
    plt.show()
    plt.clf()
    image_1 = failure_images[-1]
    plt.imshow(np.reshape(test_data[image_1[2]], (28, 28)), interpolation='nearest')
    plt.title(f"Classified: {image_1[1]} Label: {image_1[0]}")
    plt.savefig("section-1-d-img2.png")
    plt.show()
    plt.clf()


def check_w_accuracy_perceptron(w, data, labels):
    return run_check_perceptron(w, data=data, labels=labels)[0]


def run_check_perceptron(w, data, labels):
    return run_check(w, data=data, labels=labels, classifier=calc_sign)


def check_w_accuracy_sgd(w, data, labels):
    return run_check(w, data=data, labels=labels, classifier=sgd_predict)[0]


def run_check(w, data, labels, classifier):
    total = len(data)
    fails = 0.0
    classification_failures = []
    classification_success = []

    for idx, x in enumerate(data):
        sign = classifier(x, w)
        if sign != labels[idx]:
            fails += 1
            classification_failures.append((labels[idx], sign, idx))
        classification_success.append((labels[idx], sign, idx))

    accuracy = (total - fails) / total
    return accuracy, classification_failures, classification_success


def section_2_a():
    eta = find_best_eta(pow, 10, -5, 5, label="log search")
    find_best_eta(lambda x, y: x + y / 10, eta, -5, 5, label="search1")


def section_2_b():
    find_best_C(pow, 10, -5, 5, "C-LOG-SEARCH")

def section_2_c():
    T = 20000
    eta_0 = 1
    C = pow(10, -4)
    w = SGD(train_data, train_labels, C, eta_0, T)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    plt.savefig("section-2-c.png")
    plt.show()
    plt.clf()

def section_2_d():
    T = 20000
    eta_0 = 1
    C = pow(10, -4)
    w = SGD(train_data, train_labels, C, eta_0, T)
    accuracy = check_w_accuracy_sgd(w, test_data, test_labels)
    print(f"best accuracy for SGD: {accuracy}")


def find_best_eta(func, base, step_start, step_end, label):
    T = 1000
    C = 1
    eta_0_values = [func(base, i) for i in range(step_start, step_end + 1)]

    accuracy = {'eta_0': [], "avg_accuracy": []}
    for eta_0 in eta_0_values:
        results = []
        for i in range(10):
            w = SGD(train_data, train_labels, C, eta_0, T)
            results.append(check_w_accuracy_sgd(w, validation_data, validation_labels))

        accuracy["eta_0"].append(eta_0)
        accuracy["avg_accuracy"].append(np.average(results))

    plt.plot("eta_0", "avg_accuracy", data=accuracy,
             marker='o', markerfacecolor='blue', markersize=4,
             color='skyblue', linewidth=2)

    argmax = int(np.argmax(accuracy["avg_accuracy"]))
    max_for_eta = (accuracy["eta_0"][argmax], accuracy["avg_accuracy"][argmax])
    plt.annotate(f"\u03B70={max_for_eta[0]} Avg. Accuracy={max_for_eta[1]:.3f}..", xy=max_for_eta)
    # plt.xlim((eta_0_values[0], eta_0_values[-1]))
    plt.xticks(np.arange(len(eta_0_values)), eta_0_values)
    plt.xlabel("\u03B70")

    plt.xscale('log')

    plt.ylabel("Avg. Accuracy")
    plt.savefig(f"section-2-a-{label}.png")
    plt.show()
    return max_for_eta[0]


def find_best_C(func, base, step_start, step_end, label):
    T = 1000
    eta_0 = 1
    c_values = [func(base, i) for i in range(step_start, step_end + 1)]

    accuracy = {'C': [], "avg_accuracy": []}
    for C in c_values:
        results = []
        for i in range(10):
            w = SGD(train_data, train_labels, C, eta_0, T)
            results.append(check_w_accuracy_sgd(w, validation_data, validation_labels))

        accuracy["C"].append(C)
        accuracy["avg_accuracy"].append(np.average(results))

    plt.plot("C", "avg_accuracy", data=accuracy,
             marker='o', markerfacecolor='blue', markersize=4,
             color='skyblue', linewidth=2)

    argmax = int(np.argmax(accuracy["avg_accuracy"]))
    max_for_C = (accuracy["C"][argmax], accuracy["avg_accuracy"][argmax])
    plt.annotate(f"C={max_for_C[0]} Avg. Accuracy={max_for_C[1]:.3f}..", xy=max_for_C)
    # plt.xlim((eta_0_values[0], eta_0_values[-1]))
    plt.xticks(np.arange(len(c_values)), c_values)
    plt.xlabel("C")

    plt.xscale('log')

    plt.ylabel("Accuracy")
    plt.savefig(f"section-2-b-{label}.png")
    plt.show()
    return max_for_C[0]


if __name__ == "__main__":
    from sklearn.datasets.base import get_data_home

    print(get_data_home())
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper()
    # section_1_a()
    # section_1_b()
    # section_1_c()
    # section_1_d()
    #
    # section_2_a()
    # section_2_b()
    # section_2_c()
    section_2_d()
