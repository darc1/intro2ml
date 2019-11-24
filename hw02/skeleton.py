#################################
# Your name: Chen Dar
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals
from joblib import Parallel, delayed, cpu_count


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        result = np.ndarray(shape=(m, 2), dtype=float, order='F')
        x_values = np.random.rand(m)
        x_values.sort()
        for i in range(m):
            x = x_values[i]
            if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
                y = np.random.choice([0, 1], p=[0.2, 0.8])
            else:
                y = np.random.choice([0, 1], p=[0.9, 0.1])

            result[(i, 0)] = x
            result[(i, 1)] = y

        return result

    def draw_sample_intervals(self, m, k):
        """
        Plots the data as asked in (a) i ii and iii.
        Input: m - an integer, the size of the data sample.
               k - an integer, the maximum number of intervals.

        Returns: None.
        """
        values = self.sample_from_D(m)
        x = values[:, 0]
        y = values[:, 1]
        plt.plot(x, y, 'x')
        plt.xticks(np.arange(0, 1.1, step=0.2))
        plt.yticks(np.arange(0, 2))
        plt.grid()
        plt.ylim(-0.1, 1.1)

        best_intervals, error = intervals.find_best_interval(x, y, k)
        for interval in best_intervals:
            plt.plot(interval, (0.5, 0.5))

        plt.savefig("section-a.png")
        # plt.show()
        plt.clf()

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        data = {"m": [], "es": [], "ep": []}

        for m in range(m_first, m_last + 1, step):
            # true_errors = np.ndarray(T)
            # empirical_errors = np.ndarray(T)
            # for i in range(T):
            #     values = self.sample_from_D(m)
            #     x = values[:, 0]
            #     y = values[:, 1]
            #     best_intervals, emp_error = intervals.find_best_interval(x, y, k)
            #     empirical_errors[i] = emp_error / m
            #     true_error = self.calc_ep(best_intervals)
            #     true_errors[i] = true_error
            results = Parallel(n_jobs=cpu_count())(delayed(self.run_for_k_and_m)(k, m) for t in range(T))

            data["m"].append(m)
            data["es"].append(np.average([e[0] for e in results]))
            data["ep"].append(np.average([e[1] for e in results]))

        plt.plot('m', 'es', data=data, marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=0)
        plt.plot('m', 'ep', data=data, marker='o', markerfacecolor='orange', markersize=4, color='orange', linewidth=0)
        plt.xlabel('m')
        plt.legend()
        plt.savefig("section-c.png")
        # plt.show()
        plt.clf()

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """

        data = {"k": [], "es": [], "ep": []}
        samples = self.sample_from_D(m)
        # for k in range(k_first, k_last + 1):
            # values = self.sample_from_D(m)
            # x = values[:, 0]
            # y = values[:, 1]
            # h_intervals, es = intervals.find_best_interval(x, y, k)
            # ep = self.calc_ep(h_intervals)

        results = Parallel(n_jobs=cpu_count())(delayed(self.run_for_k)(k, samples) for k in range(k_first, k_last + 1, step))
        sorted(results, key=lambda item: item[2])

        for result in results:
            data["k"].append(result[2])
            data["es"].append(result[0])
            data["ep"].append(result[1])

        plt.plot('k', 'es', data=data, marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=0)
        plt.plot('k', 'ep', data=data, marker='o', markerfacecolor='orange', markersize=4, color='orange', linewidth=0)
        plt.xlabel('k')
        plt.legend()
        plt.savefig("section-d.png")
        # plt.show()
        plt.clf()
        return self.get_best_k(data["k"], data["es"])

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Runs the experiment in (d).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """

        data = {"k": [], "es": [], "ep": [], "penalty": [], "srm": []}
        samples = self.sample_from_D(m)
        # for k in range(k_first, k_last + 1):
        # values = self.sample_from_D(m)
        # x = values[:, 0]
        # y = values[:, 1]
        # h_intervals, es = intervals.find_best_interval(x, y, k)
        # ep = self.calc_ep(h_intervals)

        results = Parallel(n_jobs=cpu_count())(delayed(self.run_for_k)(k, samples) for k in range(k_first, k_last + 1, step))
        sorted(results, key=lambda item: item[2])

        for result in results:
            data["k"].append(result[2])
            data["es"].append(result[0])
            data["ep"].append(result[1])
            srm_penalty = self.srm_penalty(result[2], m, 0.1)
            data["penalty"].append(srm_penalty)
            data["srm"].append(srm_penalty + result[0])

        plt.plot('k', 'es', data=data, marker='o', markerfacecolor='blue', markersize=4, color='skyblue', linewidth=0)
        plt.plot('k', 'ep', data=data, marker='o', markerfacecolor='orange', markersize=4, color='orange', linewidth=0)
        plt.plot('k', 'penalty', data=data, marker='o', markerfacecolor='darkred', markersize=4, color='darkred', linewidth=0)
        plt.plot('k', 'srm', data=data, marker='o', markerfacecolor='green', markersize=4, color='green', linewidth=0)
        plt.xlabel('k')
        plt.legend()
        plt.savefig("section-e.png")
        # plt.show()
        plt.clf()
        return self.get_best_k(data["k"], data["srm"])

    def cross_validation(self, m, T):
        """Finds a k that gives a good test error.
        Chooses the best hypothesis based on 3 experiments.
        Input: m - an integer, the size of the data sample.
               T - an integer, the number of times the experiment is performed.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        k_values = [k for k in range(1, 11)]
        best_ks = []
        for t in range(T):
            samples = self.sample_from_D(m)
            np.random.shuffle(samples)
            holdout = samples[:m//5, :]
            train = samples[m//5:,:]
            train = train[train[:, 0].argsort()]
            sorted(train, key=lambda item: item[0])
            results = Parallel(n_jobs=cpu_count())(delayed(self.run_for_k)(k, train) for k in k_values)
            sorted(results, key=lambda item: item[2])

            ks = []
            error = []
            for result in results:
                ks.append(result[2])
                error.append(self.calc_holdout_error(holdout, result[4], m))

            best_ks.append(self.get_best_k(ks, error))

        (values, counts) = np.unique(best_ks, return_counts=True)
        best_k_idx = np.argmax(counts)
        best_overall_k = values[best_k_idx]
        print(f"best overall k is: {best_overall_k}")
        return best_overall_k


    #################################
    # Place for additional methods

    def calc_holdout_error(self, holdout_samples, h_intervals, m):
        errors = 0.0
        x = holdout_samples[:, 0]
        y = holdout_samples[:, 1]

        for i in range(len(x)):
            if self.in_intervals(x[i], h_intervals):
                if y[i] == 0:
                    errors += 1
            else:
                if y[i] == 1:
                    errors += 1

        return errors/m

    def in_intervals(self, value, interval_group):
        for interval in interval_group:
            if interval[0] <= value <= interval[1]:
                return True

        return False

    def get_best_k(self, ks, errors):
        min_error_idx = np.argmin(errors)
        best_k = ks[min_error_idx]
        print(f"best k: {best_k} error: {errors[min_error_idx]}")
        return best_k

    def srm_penalty(self, k, m, delta):
        # from theoretical questions vcdim = 2k
        return np.sqrt((8/m)*(np.log(4/delta) + 2*k*np.log((np.exp(1)*m)/k)))

    def calc_ep(self, h_intervals):
        # p1 -- x in [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]] p[Y=1|X=x]=0.8, p[Y=0|X=x]=0.2
        # p2 --  x in [[0.2, 0.4], [0.6, 0.8]] p[Y=1|X=x]=0.1, p[Y=0|X=x]=0.9
        # true error ep(h) is P[h(X) != Y]

        p1 = [0.2, 0.8]
        p2 = [0.9, 0.1]
        p1_intervals = [[0.0, 0.2], [0.4, 0.6], [0.8, 1.0]]
        p2_intervals = [[0.2, 0.4], [0.6, 0.8]]
        sum_intervals = sum([i[1] - i[0] for i in h_intervals])
        p2_overlap_with_h_complement_intervals = 0.0
        p1_overlap_with_h_intervals = 0.0
        p2_interval_value = 0.0

        for i in range(len(h_intervals)):
            p2_overlap_with_h_complement_intervals += self.overlap([p2_interval_value, h_intervals[i][0]], p2_intervals)
            p1_overlap_with_h_intervals += self.overlap(h_intervals[i], p1_intervals)
            p2_interval_value = h_intervals[i][1]

        p2_overlap_with_h_complement_intervals += self.overlap([p2_interval_value, 1], p2_intervals)

        p2_overlap_with_h_intervals = sum_intervals - p1_overlap_with_h_intervals
        p1_overlap_with_h_complement_intervals = 1 - sum_intervals - p2_overlap_with_h_complement_intervals
        return p1[0] * p1_overlap_with_h_intervals + p2[0] * p2_overlap_with_h_intervals \
               + p2[1] * p2_overlap_with_h_complement_intervals + p1[1] * p1_overlap_with_h_complement_intervals

    def overlap(self, interval, intervals_group):
        sum_overlap = 0.0
        for group_interval in intervals_group:
            sum_overlap += max(0, min(interval[1], group_interval[1]) - max(interval[0], group_interval[0]))
        return sum_overlap

    def run_for_k(self, k, samples):
        x = samples[:, 0]
        y = samples[:, 1]
        best_intervals, emp_error = intervals.find_best_interval(x, y, k)
        true_error = self.calc_ep(best_intervals)
        return emp_error/len(samples), true_error, k, len(samples), best_intervals

    def run_for_k_and_m(self,k, m):
        samples = self.sample_from_D(m)
        return self.run_for_k(k, samples)

    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.draw_sample_intervals(100, 3)
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500, 3)
