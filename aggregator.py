import os
import time
import random

from abc import ABC, abstractmethod
import numpy as np

from utils.torch_utils import *


class Aggregator(ABC):
    r""" Base class for Aggregator. `Aggregator` dictates communications between clients

    Attributes
    ----------
    clients: List[Client]

    test_clients: List[Client]

    global_learners_ensemble: List[Learner]

    sampling_rate: proportion of clients used at each round; default is `1.`

    sample_with_replacement: is True, client are sampled with replacement; default is False

    n_clients:

    n_learners:

    clients_weights:

    model_dim: dimension if the used model

    c_round: index of the current communication round

    log_freq:

    verbose: level of verbosity, `0` to quiet, `1` to show global logs and `2` to show local logs; default is `0`

    global_train_logger:

    global_test_logger:

    rng: random number generator

    np_rng: numpy random number generator

    Methods
    ----------
    __init__
    mix

    update_clients

    update_test_clients

    write_logs

    save_state

    load_state

    """

    def __init__(
            self,
            clients,
            global_learners_ensemble,
            log_freq,
            global_train_logger,
            global_test_logger,
            sampling_rate=1.,
            sample_with_replacement=False,
            test_clients=None,
            verbose=0,
            seed=None,
            *args,
            **kwargs
    ):

        rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
        self.rng = random.Random(rng_seed)
        self.np_rng = np.random.default_rng(rng_seed)

        if test_clients is None:
            test_clients = []

        self.clients = clients
        self.test_clients = test_clients

        self.global_learners_ensemble = global_learners_ensemble
        self.device = self.global_learners_ensemble.device

        self.log_freq = log_freq
        self.verbose = verbose
        self.global_train_logger = global_train_logger
        self.global_test_logger = global_test_logger

        self.model_dim = self.global_learners_ensemble.model_dim

        self.n_clients = len(clients)
        self.n_test_clients = len(test_clients)
        self.n_learners = len(self.global_learners_ensemble)

        self.clients_weights = \
            torch.tensor(
                [client.n_train_samples for client in self.clients],
                dtype=torch.float32
            )

        self.clients_weights = self.clients_weights / self.clients_weights.sum()

        self.sampling_rate = sampling_rate
        self.sample_with_replacement = sample_with_replacement
        self.n_clients_per_round = max(1, int(self.sampling_rate * self.n_clients))
        self.sampled_clients = list()

        self.c_round = 0
        self.write_logs()

    @abstractmethod
    def mix(self):
        pass

    @abstractmethod
    def update_clients(self):
        pass

    def update_test_clients(self):
        for client in self.test_clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(target=learner.model, source=self.global_learners_ensemble[learner_id].model)

        for client in self.test_clients:
            client.update_sample_weights()
            client.update_learners_weights()

    def write_logs(self):
        self.update_test_clients()

        for global_logger, clients in [
            (self.global_train_logger, self.clients),
            (self.global_test_logger, self.test_clients)
        ]:
            if len(clients) == 0:
                continue

            global_train_loss = 0.
            global_train_acc = 0.
            global_test_loss = 0.
            global_test_acc = 0.

            total_n_samples = 0
            total_n_test_samples = 0

            for client_id, client in enumerate(clients):

                train_loss, train_acc, test_loss, test_acc = client.write_logs()

                if self.verbose > 1:
                    print("*" * 30)
                    print(f"Client {client_id}..")

                    with np.printoptions(precision=3, suppress=True):
                        print("Pi: ", client.learners_weights.numpy())

                    print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.3f}%|", end="")
                    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.3f}% |")

                global_train_loss += train_loss * client.n_train_samples
                global_train_acc += train_acc * client.n_train_samples
                global_test_loss += test_loss * client.n_test_samples
                global_test_acc += test_acc * client.n_test_samples

                total_n_samples += client.n_train_samples
                total_n_test_samples += client.n_test_samples

            global_train_loss /= total_n_samples
            global_test_loss /= total_n_test_samples
            global_train_acc /= total_n_samples
            global_test_acc /= total_n_test_samples

            if self.verbose > 0:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

        if self.verbose > 0:
            print("#" * 80)

    def save_state(self, dir_path):
        """
        save the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         as `.pt` file, and `learners_weights` for each client in `self.clients` as a single numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            save_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            torch.save(learner.model.state_dict(), save_path)

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            save_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            for client_id, client in enumerate(clients):
                weights[client_id] = client.learners_ensemble.learners_weights

            np.save(save_path, weights)

    def load_state(self, dir_path):
        """
        load the state of the aggregator, i.e., the state dictionary of each `learner` in `global_learners_ensemble`
         from a `.pt` file, and `learners_weights` for each client in `self.clients` from numpy array (`.np` file).

        :param dir_path:
        """
        for learner_id, learner in enumerate(self.global_learners_ensemble):
            chkpts_path = os.path.join(dir_path, f"chkpts_{learner_id}.pt")
            learner.model.load_state_dict(torch.load(chkpts_path))

        learners_weights = np.zeros((self.n_clients, self.n_learners))
        test_learners_weights = np.zeros((self.n_test_clients, self.n_learners))

        for mode, weights, clients in [
            ['train', learners_weights, self.clients],
            ['test', test_learners_weights, self.test_clients]
        ]:
            chkpts_path = os.path.join(dir_path, f"{mode}_client_weights.npy")

            weights = np.load(chkpts_path)

            for client_id, client in enumerate(clients):
                client.learners_ensemble.learners_weights = weights[client_id]

    def sample_clients(self):
        """
        sample a list of clients without repetition

        """
        if self.sample_with_replacement:
            self.sampled_clients = \
                self.rng.choices(
                    population=self.clients,
                    weights=self.clients_weights,
                    k=self.n_clients_per_round,
                )
        else:
            self.sampled_clients = self.rng.sample(self.clients, k=self.n_clients_per_round)


class NoCommunicationAggregator(Aggregator):
    r"""Clients do not communicate. Each client work locally

    """

    def mix(self):
        self.sample_clients()
        for client in self.sampled_clients:
            client.step()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        pass


class CentralizedAggregator(Aggregator):
    r""" Standard Centralized Aggregator.
     All clients get fully synchronized with the average client.

    """

    def mix(self):
        self.sample_clients()
        sc = self.sampled_clients
        for client in sc:
            client.step()

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        # assign the updated model to all clients
        self.update_clients()

        self.c_round += 1

        if self.c_round % self.log_freq == 0:
            self.write_logs()

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )

