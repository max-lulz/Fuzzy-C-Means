import os
import random
import time
import argparse
import FuzzyCMeans
import pandas as pd
import numpy as np
import logging


def log_val(out_path, cluster_scores):
    column_labels = ["File", *list(range(2, 11))*2]

    df = pd.DataFrame(cluster_scores, columns=column_labels)
    df.to_csv(path_or_buf=out_path, index=False)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument("-d", "--DataPath", help="Path to dataset", required=True)
    args = vars(args.parse_args())
    path = args["DataPath"]

    random.seed(time.time())

    output = []

    i = 0

    for file in os.listdir(path):
        if file.endswith(".csv"):
            print("-- {} --".format(file))
            data = pd.read_csv(os.path.join(path, file), sep=",", header=None)
            data = np.delete(data.values, 20, 1)

            dataset_score = [None] * 19
            dataset_score[0] = file

            for k in range(2, 11):
                print(k)
                fcm = FuzzyCMeans.FuzzyCMeans(k, data)
                fcm.cluster_points()
                # print(fcm.get_cluster_scores())

                silhouette, davies_bouldin = fcm.get_cluster_scores()
                dataset_score[k - 1] = silhouette
                dataset_score[k - 1 + 9] = davies_bouldin

            output.append(dataset_score)

    # print(output)
    log_val("out.csv", output)
