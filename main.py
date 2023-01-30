import time

import networkx as nx

from admm_solver import NetworkLassoRunner
from utils.admm_utils import get_local_data
from utils.plots import make_plot, INPUT_DIM
from utils.utils import *
from utils.constants import *
from utils.args import *
from datasets import *

from torch.utils.tensorboard import SummaryWriter


def init_clients(args_, root_path, logs_dir):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_dir: path to logs root
    :return: List[Client]
    """
    print("===> Building data iterators..")
    train_iterators, val_iterators, test_iterators = \
        get_loaders(
            type_=args_.experiment,
            root_path=root_path,
            batch_size=args_.bz,
            is_validation=args_.validation,
            is_binary=args_.output_dimension == 1
        )

    print("===> Initializing clients..")
    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        learners_ensemble = \
            get_learners_ensemble(
                name=args_.experiment,
                n_learners=args_.n_learners,
                device=args_.device,
                optimizer_name=args_.optimizer,
                scheduler_name=args_.lr_scheduler,
                initial_lr=args_.lr,
                input_dim=INPUT_DIM[args_.experiment],
                output_dim=args_.output_dimension,
                n_rounds=args_.n_rounds,
                seed=args_.seed,
                mu=args_.mu
            )

        logs_path = os.path.join(logs_dir, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)

    return clients_


def run_experiment(args_):
    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_dir" in args_:
        logs_dir = args_.logs_dir
    else:
        logs_dir = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients = init_clients(args_, root_path=os.path.join(data_dir, "train"), logs_dir=os.path.join(logs_dir, "train"))

    print("==> Test Clients initialization..")
    test_clients = init_clients(args_, root_path=os.path.join(data_dir, "test"),
                                logs_dir=os.path.join(logs_dir, "test"))

    logs_path = os.path.join(logs_dir, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_dir, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            name=args_.experiment,
            n_learners=args_.n_learners,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=INPUT_DIM[args_.experiment],
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu
        )

    aggregator_type = 'centralized'

    aggregator = \
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    print("Training..")
    pbar = tqdm(total=args_.n_rounds)
    current_round = 0
    while current_round <= args_.n_rounds:

        aggregator.mix()

        if aggregator.c_round != current_round:
            pbar.update(1)
            current_round = aggregator.c_round

    if "save_dir" in args_:
        save_dir = os.path.join(args_.save_dir)

        os.makedirs(save_dir, exist_ok=True)
        aggregator.save_state(save_dir)


def run_admm_experiment(args):
    c = 0.75
    num_features = INPUT_DIM[args.experiment]

    G = build_graph(100)
    runner = NetworkLassoRunner(G)
    inputs, targets = get_dataset(args.experiment, is_tensor=False)
    datasets = [None] * 100
    datasets_test = [None] * 100
    for node_id in range(100):
        datasets[node_id] = get_local_data(node_id, args.experiment, inputs, targets)
        datasets_test[node_id] = get_local_data(node_id, args.experiment, inputs, targets, True)
    w, lambs, accuracies = runner.run(num_features, args.n_rounds, datasets, datasets_test, c)
    return w, lambs, accuracies


def build_graph(num_nodes):
    print("Building a simulated MEC topology for " + str(num_nodes) + " nodes.")
    if num_nodes == 100:
        neighbours = [[73, 38, 88, 1, 42], [75, 64, 88, 12, 67], [77, 60, 56, 17, 13], [30, 75, 74, 93, 64],
                      [90, 27, 51, 47, 82], [96, 16, 55, 95, 28], [12, 7, 88, 42, 64], [6, 12, 42, 88, 64],
                      [48, 18, 80, 47, 61], [11, 33, 85, 44, 87], [34, 15, 58, 50, 43], [9, 33, 85, 84, 87],
                      [88, 6, 75, 64, 1], [81, 60, 2, 17, 56], [30, 3, 1, 74, 39], [34, 50, 10, 32, 91],
                      [28, 5, 87, 95, 96], [77, 60, 2, 13, 1], [48, 8, 24, 47, 80], [66, 79, 7, 6, 63],
                      [89, 81, 13, 50, 92], [25, 28, 95, 87, 49], [52, 54, 26, 53, 71], [43, 97, 62, 58, 10],
                      [44, 25, 18, 8, 28], [44, 24, 49, 21, 28], [76, 71, 54, 53, 86], [90, 4, 51, 82, 47],
                      [16, 87, 5, 95, 44], [59, 99, 82, 47, 80], [3, 74, 14, 75, 88], [68, 23, 97, 62, 43],
                      [91, 45, 35, 15, 60], [11, 9, 85, 44, 18], [15, 10, 50, 58, 32], [32, 60, 91, 93, 45],
                      [86, 96, 87, 5, 16], [5, 16, 96, 36, 28], [42, 1, 67, 0, 12], [14, 45, 46, 0, 91],
                      [42, 67, 38, 1, 12], [50, 81, 20, 62, 89], [40, 12, 6, 38, 88], [58, 62, 10, 23, 41],
                      [24, 25, 49, 28, 16], [91, 32, 46, 39, 14], [45, 1, 14, 78, 39], [80, 51, 59, 48, 82],
                      [47, 80, 61, 8, 59], [29, 59, 44, 25, 80], [81, 20, 15, 41, 13], [82, 47, 80, 4, 79],
                      [54, 26, 86, 57, 36], [71, 26, 76, 86, 54], [52, 86, 36, 26, 71], [5, 96, 95, 86, 76],
                      [2, 60, 77, 93, 13], [54, 52, 36, 98, 86], [43, 10, 34, 62, 50], [29, 99, 47, 80, 51],
                      [13, 2, 17, 77, 56], [80, 48, 99, 47, 79], [43, 97, 58, 41, 10], [65, 72, 38, 6, 42],
                      [75, 12, 1, 88, 3], [63, 79, 61, 48, 72], [19, 88, 6, 7, 12], [1, 40, 38, 64, 75],
                      [23, 97, 43, 62, 58], [94, 70, 22, 52, 57], [94, 69, 22, 52, 57], [53, 26, 86, 76, 54],
                      [63, 42, 38, 40, 65], [0, 38, 42, 88, 12], [30, 3, 93, 14, 75], [64, 88, 12, 1, 3],
                      [26, 96, 5, 55, 86], [2, 17, 60, 64, 3], [46, 45, 39, 91, 32], [51, 99, 80, 47, 4],
                      [47, 99, 51, 48, 59], [13, 50, 20, 60, 2], [51, 47, 80, 29, 90], [85, 84, 99, 61, 49],
                      [83, 85, 99, 61, 59], [83, 84, 11, 33, 9], [36, 96, 95, 98, 5], [16, 28, 95, 36, 96],
                      [12, 75, 64, 6, 1], [92, 20, 81, 50, 13], [27, 4, 51, 82, 47], [32, 45, 60, 35, 14],
                      [89, 20, 81, 13, 41], [3, 74, 30, 56, 60], [69, 70, 22, 52, 57], [96, 5, 86, 16, 87],
                      [5, 86, 95, 36, 55], [62, 23, 43, 58, 41], [86, 87, 36, 71, 95], [80, 59, 79, 47, 51]]
    elif num_nodes == 150:
        neighbours = []
    elif num_nodes == 200:
        neighbours = []
    elif num_nodes == 250:
        neighbours = []
    elif num_nodes == 300:
        neighbours = []
    elif num_nodes == 350:
        neighbours = []
    else:
        neighbours = []

    G = nx.Graph()
    for node_id in range(num_nodes):
        G.add_node(node_id)
        for neighbour_id in neighbours[node_id]:
            G.add_edge(node_id, neighbour_id, weight=1)
    return G


if __name__ == "__main__":
    args = parse_args()
    if args.method != "sadmm":
        run_experiment(args)
    else:
        reulst = run_admm_experiment(args)
        print(reulst)

    path = os.getcwd() + '\\' + time.strftime("%H%M-%d%m%Y", time.localtime())
    if not os.path.exists(path):
        os.makedirs(path)
    make_plot("./logs/" + args.experiment, "Train/Metric", path)
    make_plot("./logs/" + args.experiment, "Train/Loss", path)
    make_plot("./logs/" + args.experiment, "Test/Loss", path)
    make_plot("./logs/" + args.experiment, "Test/Metric", path)
    print("Experiment Complete")
