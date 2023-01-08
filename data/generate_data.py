import os
import argparse
import pickle

import pandas as pd

from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from data.utils import split_dataset_by_labels, pathological_non_iid_split


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str
    )
    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; '
             'default is 0.2',
        type=float,
        default=0.2)
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 0.2;',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--val_frac',
        help='fraction in validation set (from train set); default: 0.0;',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--test_tasks_frac',
        help='fraction of tasks / clients not participating to the training; default is 0.0',
        type=float,
        default=0.0
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )
    parser.add_argument(
        "--classes",
        help='The number of categories in the dataset',
        type=int,
        default=2
    )

    return parser.parse_args()


class ExperimentDataset(Dataset):
    def __init__(self, data):
        self.data = data.iloc[:, :-1]
        self.targets = data.iloc[:, -1]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data.iloc[idx]
        y = self.targets.iloc[idx]
        return x, y


if __name__ == '__main__':
    args = parse_args()
    file_dir = args.experiment + "/raw/data_set"
    save_path = args.experiment + "/all_data/"
    assert os.path.isdir(file_dir), "There is no datasets"
    all_csv_list = os.listdir(file_dir)
    all_data = None
    for csv_file in all_csv_list:
        # cols = list(pd.read_csv(os.path.join(file_dir, csv_file), nrows=1, header=None))
        data = pd.read_csv(os.path.join(file_dir, csv_file), header=None)
        if csv_file == all_csv_list[0]:
            all_data = data
        else:
            all_data = pd.concat([all_data, data], ignore_index=True)

    # all_data = all_data.sample(frac=1).reset_index(drop=True)
    dataset = ExperimentDataset(all_data)

    if args.pathological_split:
        clients_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=args.classes,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=args.classes,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed,
            )

    if args.test_tasks_frac > 0:
        train_clients_indices, test_clients_indices = \
            train_test_split(clients_indices, test_size=args.test_tasks_frac, random_state=args.seed)
    else:
        train_clients_indices, test_clients_indices = clients_indices, []

    os.makedirs(os.path.join(save_path, "train"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "test"), exist_ok=True)

    for mode, clients_indices in [('train', train_clients_indices), ('test', test_clients_indices)]:
        for client_id, indices in enumerate(clients_indices):
            client_path = os.path.join(save_path, mode, "task_{}".format(client_id))
            os.makedirs(client_path, exist_ok=True)

            train_indices, test_indices = \
                train_test_split(
                    indices,
                    train_size=args.tr_frac,
                    random_state=args.seed
                )

            if args.val_frac > 0:
                train_indices, val_indices = \
                    train_test_split(
                        train_indices,
                        train_size=1. - args.val_frac,
                        random_state=args.seed
                    )

                save_data(val_indices, os.path.join(client_path, "val.pkl"))

            save_data(train_indices, os.path.join(client_path, "train.pkl"))
            save_data(test_indices, os.path.join(client_path, "test.pkl"))

    print("completed")
