from utils.utils import *
from utils.plots import INPUT_DIM
from utils.constants import *
from utils.args import *

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


def getTrainDQNServer(args_):
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

    aggregator_type = 'trainDQN'

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
            seed=args_.seed,
            config=args_
        )
    return aggregator


if __name__ == '__main__':
    config = parse_args()
    if not os.path.exists("./model"):
        os.makedirs("./model", exist_ok=True)
    if not os.path.exists("./output"):
        os.makedirs("./output", exist_ok=True)
    fn = config.rewards_log
    print("Reards logs written to:", fn)
    with open(fn, 'w') as f:
        f.write('Episode,Reward,Round,Accuracy\n')
    aggregator = getTrainDQNServer(config)
    aggregator.profile_all_clients()
    for i_episode in range(aggregator.episode):
        print()
        t_start = time.time()
        # calculate the epsilon value for the current episode
        epsilon_current = aggregator.config.epsilon_initial * pow(aggregator.config.epsilon_decay, i_episode)
        epsilon_current = max(aggregator.config.epsilon_min, epsilon_current)

        total_reward, com_round, final_acc = aggregator.train_episode(i_episode + 1, epsilon_current)

        t_end = time.time()
        print("Episode: {}/{}, total_reward: {}, com_round: {}, final_acc: {:.4f}, time: {:.2f} s"
              .format(i_episode+1, aggregator.episode, total_reward, com_round, final_acc, t_end - t_start))
        with open(fn, 'a') as f:
            f.write('{},{},{},{}\n'.format(i_episode, total_reward, com_round, final_acc))
        # save trained model to h5 file
        model_fn = aggregator.config.saved_model + '_' + str(i_episode) + '.h5'
        aggregator.dqn_model.save(model_fn)
        print("DQN model saved to:", model_fn)

    print("\nTraining finished!")
