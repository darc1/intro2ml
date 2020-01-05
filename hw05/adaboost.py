#################################
# Your name: Chen Dar
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib.
from matplotlib import pyplot as plt
import numpy as np
from process_data import parse_data

np.random.seed(7)


def run_adaboost(X_train, y_train, T):
    """
    Returns: 

        hypotheses : 
            A list of T tuples describing the hypotheses chosen by the algorithm. 
            Each tuple has 3 elements (h_pred, h_index, h_theta), where h_pred is 
            the returned value (+1 or -1) if the count at index h_index is <= h_theta.

        alpha_vals : 
            A list of T float values, which are the alpha values obtained in every 
            iteration of the algorithm.
    """
    num_train_samples = len(X_train)
    num_features = len(X_train[0])
    weights = np.full(shape=(num_train_samples,), fill_value=1 / num_train_samples)
    alpha_vals = []
    hypotheses = []

    for t in range(T):
        best_hypotheses = find_best_hypotheses(X_train, num_features, weights, y_train)
        total_error = calc_hypotheses_error(X_train, y_train, best_hypotheses ,weights)
        alpha = calc_amount_of_say(total_error)
        weights = adjust_weights(X_train, y_train, best_hypotheses, weights, alpha, total_error)
        # print(f"found best h={best_hypotheses}, error: {total_error}, alpha:{alpha} sum adjusted weights={sum(weights)} for t={t}")

        alpha_vals.append(alpha)
        hypotheses.append(best_hypotheses)

    return hypotheses, alpha_vals
        ##############################################


# You can add more methods here, if needed.

def find_best_hypotheses(X_train, num_features, weights, y_train):
    best_error_1 = float('inf')
    best_error_2 = float('inf')
    best_feature_1 = -1
    best_feature_2 = -1
    threshold_1 = -1.0
    threshold_2 = -1.0
    for feature in range(num_features):

        feature_values = sorted(zip(X_train[:, feature], y_train, weights), key=lambda v: v[0])
        # print(feature_values)

        curr_error_1 = 0
        curr_error_2 = 0
        num_feature_values = len(feature_values)
        for i in range(num_feature_values):
            if feature_values[i][1] == 1:
                curr_error_1 += feature_values[i][2]
            else:
                curr_error_2 += feature_values[i][2]

        if curr_error_1 < best_error_1:
            best_error_1 = curr_error_1
            threshold_1 = feature_values[0][0] - 1
            best_feature_1 = feature

        if curr_error_2 < best_error_2:
            best_error_2 = curr_error_2
            threshold_2 = feature_values[0][0] - 1
            best_feature_2 = feature

        feature_values.append((feature_values[num_feature_values - 1][0] + 1, 0, 0))
        for i in range(num_feature_values):
            curr_error_1 = curr_error_1 - feature_values[i][1] * feature_values[i][2]
            if curr_error_1 < best_error_1 and feature_values[i][0] != feature_values[i + 1][0]:
                best_error_1 = curr_error_1
                threshold_1 = (feature_values[i][0] + feature_values[i + 1][0]) / 2
                best_feature_1 = feature

            curr_error_2 = curr_error_2 + feature_values[i][1] * feature_values[i][2]
            if curr_error_2 < best_error_2 and feature_values[i][0] != feature_values[i + 1][0]:
                best_error_2 = curr_error_2
                threshold_2 = (feature_values[i][0] + feature_values[i + 1][0]) / 2
                best_feature_2 = feature

    if best_error_1 < best_error_2:
        return 1, best_feature_1, threshold_1
    return -1, best_feature_2, threshold_2


def calc_hypotheses_error(X_train, y_train, h, weights):
    error = 0.0
    for i in range(len(X_train)):
        if run_hypotheses(h, X_train[i]) != y_train[i]:
            error += weights[i]
    return error


def calc_amount_of_say(error):
    return 0.5 * np.log((1 - error) / error)


def adjust_weights(X_train, y_train, h, weights, amount_of_say, total_error):
    for i in range(len(weights)):
        exp = np.exp(-1 * y_train[i] * run_hypotheses(h, X_train[i]) * amount_of_say)
        weights[i] = (weights[i] * exp)/(2*np.sqrt(total_error*(1 - total_error)))

    return weights


def run_hypotheses(h, x):
    if x[h[1]] <= h[2]:
        return h[0]
    return -1 * h[0]

def calc_avg_exp_loss(X, Y, T, alphas, hs):
    sum = 0.0
    m = len(X)
    for i in range(m):
        a = 0.0
        for j in range(T):
            a += alphas[j]*run_hypotheses(hs[j], X[i])
        sum += np.exp(-1*Y[i]*a)

    return sum/m


##############################################


def main():
    data = parse_data()
    if not data:
        return
    (X_train, y_train, X_test, y_test, vocab) = data

    T = 10
    hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # print(f"{[ (vocab[h[1]], h[1]) for h in hypotheses] }")

    # T = 80
    # hypotheses, alpha_vals = run_adaboost(X_train, y_train, T)
    # print(f"{[ (vocab[h[1]], h[1]) for h in hypotheses] }")
    #
    # train_exp_loss = []
    # test_exp_loss = []
    # for t in range(T):
    #     train_exp_loss.append(calc_avg_exp_loss(X_train, y_train, t, alpha_vals, hypotheses))
    #     test_exp_loss.append(calc_avg_exp_loss(X_test, y_test, t, alpha_vals, hypotheses))
    #
    # plt.plot( [t for t in range(T)], train_exp_loss, marker='x', color='red')
    # plt.plot( [t for t in range(T)], test_exp_loss, marker='o', color='green')
    # plt.legend(['train exp loss', 'test exp loss'], loc='upper left')
    # plt.savefig("section-c.png")
    # plt.show()

    ##############################################
    # You can add more methods here, if needed.

    ##############################################


if __name__ == '__main__':
    main()
