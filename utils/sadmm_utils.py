import os

import numpy as np

from datasets import ExperimentDataset

np.random.seed(10)


def get_value(data, node_id, neighbour_id):
    try:
        neighbour_data = data[(node_id, neighbour_id)]
        val = neighbour_data[0]
    except KeyError:
        neighbour_data = data[(neighbour_id, node_id)]
        val = neighbour_data[1]

    return val


def calculate_consensus(z, edges):
    num_edges = len(z)
    cons = 0
    for edge in edges:
        z_edge = z[edge]
        if np.all(z_edge[0] == z_edge[1]):
            cons += 1
    return cons / float(num_edges)


def calculate_accuracy(x, num_nodes, datasets_test, logger=None, iterate_num=None):
    total_correct_preds = 0
    num_total_test_samples = 0

    for node_id in range(num_nodes):
        X_test, y_test = datasets_test[node_id]

        num_test_samples = y_test.shape[0]
        num_total_test_samples += num_test_samples
        a = np.array(x[:, node_id])
        a = a.reshape(1, a.shape[0])
        y_pred = np.sign(np.dot(X_test, a.T)).flatten()
        correct_preds = int(np.sum(np.abs(y_pred + y_test)) / 2)
        total_correct_preds += correct_preds

    if iterate_num is None and logger is None:
        global_correct_preds = total_correct_preds / float(num_total_test_samples)
        print("Global Accuracy: " + str(global_correct_preds))
        return global_correct_preds
    else:
        global_acc = total_correct_preds / float(num_total_test_samples)
        print("Iteration: " + str(iterate_num) + ", Accuracy: " + str(global_acc))
        logger.add_scalar("Test/Metric", global_acc, iterate_num)


def get_local_data(node_id, dataset_name, inputs, targets, is_test=False):
    dataset_name = "./data/" + dataset_name + "/all_data/train/task_" + str(node_id)
    if is_test:
        path = os.path.join(dataset_name, f"test.pkl")
    else:
        path = os.path.join(dataset_name, f"train.pkl")
    local_dataset = ExperimentDataset(path, inputs, targets)
    return local_dataset.data, local_dataset.targets
