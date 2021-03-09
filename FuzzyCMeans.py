import numpy as np
import pandas as pd
import random
from sklearn import metrics
import time
import logging


def func_time(func):
    def timer(*args, **kwargs):
        init_time = time.time()
        func_return_val = func(*args, **kwargs)
        logging.debug(time.time() - init_time)

        return func_return_val

    return timer


class FuzzyCMeans:

    def __init__(self, C, dataset):

        self.num_C = C
        self.fuzziness = 2

        self.dataset = dataset
        self.num_pts = dataset.shape[0]
        self.num_feat = dataset.shape[1]

        self.cluster_centre = np.empty((self.num_C, self.num_feat))
        self.membership_val = np.empty((self.num_pts, self.num_C))
        self.cluster_labels = {}

        self.eps = 0.00001

        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    def get_distance(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    @func_time
    def get_membership(self):

        pt_to_cluster = np.empty(self.num_C)
        inv_fuzzy_sum = 0

        def get_pt_membership(curr_cluster):

            dist_curr_cluster = pt_to_cluster[cluster]
            inv_membership = (dist_curr_cluster ** 2 / (self.fuzziness - 1)) * inv_fuzzy_sum

            return 1 / inv_membership

        for pt in range(self.num_pts):

            inv_fuzzy_sum = 0
            is_centroid = False

            for cluster in range(self.num_C):
                pt_to_cluster[cluster] = self.get_distance(self.dataset[pt], self.cluster_centre[cluster])

                if pt_to_cluster[cluster] == 0.0:
                    is_centroid = True

                    self.membership_val[pt] = np.zeros(self.num_C)
                    self.membership_val[pt][cluster] = 1.0

                    break

                inv_fuzzy_sum += (1 / pt_to_cluster[cluster]) ** (2 / (self.fuzziness - 1))

            if is_centroid:
                continue

            for cluster in range(self.num_C):
                self.membership_val[pt][cluster] = get_pt_membership(cluster)

    def assign_labels(self):

        self.cluster_labels = {}

        for pt in range(self.num_pts):
            pt_label = self.membership_val[pt].argmax()
            self.cluster_labels.setdefault(pt_label, []).append(pt)

    def get_new_centres(self):

        for cluster, points in self.cluster_labels.items():
            sum_weight = 0
            sum_weighted_pts = np.zeros(self.num_feat)

            for pt in points:
                weight = self.membership_val[pt][cluster] ** self.fuzziness

                sum_weight += weight
                sum_weighted_pts += weight * self.dataset[pt]

            self.cluster_centre[cluster] = sum_weighted_pts / sum_weight

    def get_random_centres(self):

        rand_pts = random.sample(range(self.num_pts), self.num_C)

        for i, pt in enumerate(rand_pts):
            self.cluster_centre[i] = self.dataset[pt]

    def __silhouette(self, labels):

        def intra_cluster(point):
            return self.get_distance(self.dataset[point], self.cluster_centre[labels[point]])

        def inter_cluster(point):
            dist = 1e15

            for cluster in range(self.num_C):
                if labels[point] != cluster:
                    dist = min(dist, self.get_distance(self.dataset[point], self.cluster_centre[cluster]))

            return dist

        silh_coeff = 0

        for pt in range(self.num_pts):
            ai = intra_cluster(pt)
            bi = inter_cluster(pt)

            silh_coeff += (bi - ai) / max(bi, ai)

        return silh_coeff / self.num_pts

    def cluster_points(self):

        self.get_random_centres()
        self.get_membership()
        num_iter = 0

        while num_iter <= 100:

            mem_prev = self.membership_val.copy()

            self.assign_labels()
            self.get_new_centres()
            self.get_membership()

            if self.get_distance(mem_prev, self.membership_val) < self.eps:
                break

            num_iter += 1

    def get_cluster_scores(self):
        labels = np.empty(self.num_pts, dtype=np.uint8)
        for label in self.cluster_labels:
            for pt in self.cluster_labels[label]:
                labels[pt] = label

        return metrics.silhouette_score(self.dataset, labels), metrics.davies_bouldin_score(self.dataset, labels)


if __name__ == '__main__':

    random.seed(time.time())

    data = pd.read_csv("Data/1.csv", sep=",", header=None)
    data = np.delete(data.values, 20, 1)

    for ind in range(2, 11):
        aa = FuzzyCMeans(ind, data)
        aa.cluster_points()

        print(aa.get_cluster_scores())
