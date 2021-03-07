import numpy as np
import random

class Fuzzy:

    def __init__(self, C, dataset):

        self.num_C = C
        self.fuzziness = 2

        self.dataset = dataset
        self.num_pts = dataset.shape[0]
        self.num_feat = dataset.shape[1]

        self.cluster_centre = np.empty((self.num_C, self.num_feat))
        self.membership_val = np.empty((self.num_pts, self.num_C))
        self.cluster_labels = {}
        self.cluster_labels.setdefault([])

    def get_distance(self, point_a, point_b):
        return np.linalg.norm(point_a - point_b)

    def get_val_objectivefunc(self):

    def get_membership(self):

        def get_pt_membership(curr_pt, curr_cluster):

            if np.array_equal(curr_pt, curr_cluster):
                return 1

            inv_membership = 0
            dist_curr_cluster = self.get_distance(curr_pt, curr_cluster)

            for cluster_cnt in self.cluster_centre:

                inv_membership += (dist_curr_cluster/self.get_distance(curr_pt, cluster_cnt)) \
                                  ** (2/(self.fuzziness-1))                                                             # optimise

            return 1/inv_membership

        for pt in range(self.num_pts):
            for cluster in range(self.num_C):
                self.membership_val[pt][cluster] = get_pt_membership(self.dataset[pt], self.cluster_centre[cluster])

    def get_new_centres(self):

        for cluster, points in self.cluster_labels:
            sum_weight = 0
            sum_weighted_pts = np.zeros(self.num_feat)

            for pt in points:
                weight = self.membership_val[pt][cluster] ** self.fuzziness

                sum_weight += weight
                sum_weighted_pts += weight * self.dataset[pt]

            self.cluster_centre[cluster] = sum_weighted_pts/sum_weight


    def assign_labels(self):

        self.cluster_labels = {}

        for pt in range(self.num_pts):
            pt_label = self.membership_val[pt].argmax()
            self.cluster_labels.setdefault(pt_label, []).append(pt)

    def get_random_centres(self):

        rand_pts = random.sample(range(self.num_pts), self.num_C)

        for i, pt in enumerate(rand_pts):
            self.cluster_centre[i] = self.dataset[pt]

    def cluster_points(self):

        self.get_random_centres()
        # self.get_membership()




