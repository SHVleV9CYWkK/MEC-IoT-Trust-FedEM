import os
import time
import random
import pickle as pk

from sklearn.decomposition import PCA
from tensorflow import keras
from collections import deque
from keras.layers import Dense, Input, Dropout, ReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import huber_loss
from abc import ABC, abstractmethod
import numpy as np
from decimal import Decimal

from models import DQNModule
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

    def write_logs(self, display=True):
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

                if self.verbose > 1 and display:
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

            if self.verbose > 0 and display:
                print("+" * 30)
                print("Global..")
                print(f"Train Loss: {global_train_loss:.3f} | Train Acc: {global_train_acc * 100:.3f}% |", end="")
                print(f"Test Loss: {global_test_loss:.3f} | Test Acc: {global_test_acc * 100:.3f}% |")
                print("+" * 50)

            global_logger.add_scalar("Train/Loss", global_train_loss, self.c_round)
            global_logger.add_scalar("Train/Metric", global_train_acc, self.c_round)
            global_logger.add_scalar("Test/Loss", global_test_loss, self.c_round)
            global_logger.add_scalar("Test/Metric", global_test_acc, self.c_round)

            return Decimal(global_test_acc).quantize(Decimal("0.00")), self.c_round

        if self.verbose > 0 and display:
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
            return self.write_logs()

    def update_clients(self):
        for client in self.clients:
            for learner_id, learner in enumerate(client.learners_ensemble):
                copy_model(learner.model, self.global_learners_ensemble[learner_id].model)

                if callable(getattr(learner.optimizer, "set_initial_params", None)):
                    learner.optimizer.set_initial_params(
                        self.global_learners_ensemble[learner_id].model.parameters()
                    )


class CentralizedAggregatorWithDQN(CentralizedAggregator):
    def __init__(self,
                 clients,
                 global_learners_ensemble,
                 log_freq,
                 global_train_logger,
                 global_test_logger,
                 train_dqn=False,
                 *args,
                 **kwargs):
        super().__init__(clients,
                         global_learners_ensemble,
                         log_freq,
                         global_train_logger,
                         global_test_logger,
                         *args,
                         **kwargs)
        self.pca = None
        self.pca_weights_clientserver = None

    def sample_clients(self):
        # Select devices to participate in current round
        clients_per_round = self.n_clients_per_round
        print('self.pca_weights_clientserver.shape:', self.pca_weights_clientserver.shape)

        # calculate state using the pca model transformed weights
        state = self.pca_weights_clientserver.flatten()
        state = state.tolist()

        # use dqn model to select top k devices
        q_values = self.dqn_model.predict([state])[0]
        print("q_values: ", q_values)

        # select top k index based on the q_values
        top_k_index = np.argsort(q_values)[-clients_per_round:]
        print("top_k_index: ", top_k_index)

        self.sampled_clients = [self.clients[idx] for idx in top_k_index]

    def load_pca(self, pca_model_fn):
        print("Load saved PCA model from:", pca_model_fn)
        self.pca = pk.load(open(pca_model_fn, 'rb'))
        print("PCA model loaded.")

    def load_dqn_model(self, trained_model):
        self.dqn_model = keras.models.load_model(trained_model)
        print("Loaded trained DQN model from:", trained_model)

    def profile_all_clients(self):
        # all clients send updated weights to server, the server will do FedAvg
        # And then run  PCA and store the transformed weights

        print("Start profiling all clients...")

        # Perform profiling on all clients
        # clients_weights_pca, server_weights_pca = self.profiling(self.clients, train_dqn)
        for client in self.clients:
            client.step()

        clients_weights_pca = self.pca.transform(np.array([flatten_weights(self.clients_weights)]))

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        self.update_clients()
        server_weights_pca = self.pca.transform(np.array([flatten_weights(self.global_learners_ensemble[0].model)]))

        # save the initial pca weights for each client + server
        self.pca_weights_clientserver_init = np.vstack((clients_weights_pca, server_weights_pca))
        print("shape of self.pca_weights_clientserver_init: ", self.pca_weights_clientserver_init.shape)

        # save a copy for later update in DQN training episodes
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

        print('self.pca_weights_clientserver.shape:', self.pca_weights_clientserver.shape)


@staticmethod
def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())
    return np.array(weight_vecs)


class DQNTrainServer(Aggregator, ABC):
    """Federated learning server that uses Double DQN for device selection."""

    def __init__(self, config, clients, global_learners_ensemble, log_freq, global_train_logger,
                 global_test_logger, *args, **kwargs):
        super().__init__(clients, global_learners_ensemble, log_freq, global_train_logger, global_test_logger, *args,
                         **kwargs)
        self.config = config
        self.memory = deque(maxlen=self.config.memory_size)
        self.nA = len(self.clients)
        self.model = None
        self.episode = self.config.episode
        self.max_steps = self.config.max_steps
        self.target_update = self.config.target_update
        self.batch_size = self.config.batch_size
        self.gamma = self.config.gamma
        # number of components to use for PCA, notice here pca_n_components should be smaller than the total number of clients!!!
        self.pca_n_components = min(100, len(self.clients))
        self.pca = None

        # self.dqn_model = self._build_model()
        # self.target_model = self._build_model()

        self.dqn_model = self._build_model2()
        self.target_model = self._build_model2()

        self.pca_weights_clientserver_init = None
        self.pca_weights_clientserver = None

        print("nA =", self.nA)
        # self.total_steps = 0

    def _build_model(self):
        layers = [128]  # hidden layers

        # (all clients weight + server weight) * pca_n_components, flattened to 1D
        input_size = (len(self.clients) + 1) * self.pca_n_components

        states = Input(shape=(input_size,))
        z = states
        for l in layers:
            z = Dense(l, activation='linear')(z)

        q = Dense(len(self.clients), activation='linear')(
            z)  # here use linear activation function to predict the q values for each action/client

        model = Model(inputs=[states], outputs=[q])
        model.compile(optimizer=Adam(lr=self.config.learning_rate), loss=huber_loss)

        return model

    def _build_model2(self):

        # use the 2layer MLP torch model in fl-lottery/rl/agent.py
        # https://github.com/iQua/fl-lottery/blob/360d9c2d54c12e2631ac123a4dd5ac9184d913f0/rl/agent.py

        layers = [128]  # hidden layers
        l1 = layers[0]

        # (all clients weight + server weight) * pca_n_components, flattened to 1D
        input_size = (len(self.clients) + 1) * self.pca_n_components

        states = Input(shape=(input_size,))

        z = Dense(l1, activation='linear')(states)
        z = Dropout(0.5)(z)
        z = ReLU()(z)
        q = Dense(len(self.clients), activation='linear')(z)

        model = Model(inputs=[states], outputs=[q])
        model.compile(optimizer=Adam(lr=self.config.learning_rate), loss=huber_loss)

        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.dqn_model.get_weights())

    def epsilon_greedy(self, state, epsilon_current):

        nA = self.nA
        epsilon = epsilon_current  # the probability of choosing a random action
        action_probs = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(self.dqn_model.predict([state])[0])
        action_probs[best_action] += (1 - epsilon)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def create_greedy_policy(self):

        def policy_fn(state):
            return np.argmax(self.dqn_model.predict([state])[0])

        return policy_fn

    def dqn_round(self, random=False, action=0):
        # default: select the

        # import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        if random:
            sample_clients = self.sample_clients()
            print("randomly select clients:", sample_clients)
        else:
            sample_clients = self.dqn_selection(action)
            print("dqn select clients:", sample_clients)

        sample_clients_ids = [inx for inx, client in enumerate(sample_clients)]

        for client in self.clients:
            client.step()

        # client weights pca
        clients_weights_pca = self.pca.transform(np.array([flatten_weights(self.clients_weights)]))

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        self.update_clients()
        self.save_model(self.global_learners_ensemble[0].model, "/")
        server_weights_pca = self.pca.transform(np.array([flatten_weights(self.global_learners_ensemble[0].model)]))

        accuracy = self.write_logs()
        # update the weights of the selected devices and server to corresponding client id
        # return next_state
        for i in range(len(sample_clients_ids)):
            self.pca_weights_clientserver[sample_clients_ids[i]] = clients_weights_pca[i]

        self.pca_weights_clientserver[-1] = server_weights_pca[0]

        next_state = self.pca_weights_clientserver.flatten()
        next_state = next_state.tolist()

        # testing accuracy, updated pca_weights_clientserver
        return accuracy, next_state

    def dqn_reset_state(self):

        # randomly select k devices to conduct 1 round of FL to reset the states
        # only update the weights of the selected devices in self.pca_weights_clientserver_init

        # copy over again
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

        # randomly select k devices, update the weights of the selected devices and server to get next_state
        accuracy, new_state = self.dqn_round(random=True)  # updated self.pca_weights_clientserver
        self.prev_accuracy = accuracy

        return new_state

    def choose_action(self, state):

        # predict the q values for each action given the current state
        print("DQN choose action")
        q_values = self.dqn_model.predict([state], verbose=0)[0]

        # print("q_values:", q_values)
        # use a softmax function to convert the q values to probabilities
        probs = np.exp(q_values) / np.sum(np.exp(q_values))
        # print("probs:", probs)

        # add small value to each probability to avoid 0 probability
        # probs = probs + 0.000001

        # choose an action based on the probabilities
        action = np.random.choice(self.nA, p=probs)

        return action

    def train_episode(self, episode_ct, epsilon_current):

        # must reload the initial model for each episode
        self.load_model()  # save initial global model

        # reset the state at beginning of each episode, randomly select k devices to reset the states
        state = self.dqn_reset_state()  # ++ reset the state at beginning of each episode, randomly select k devices to reset the states

        total_reward = 0
        com_rounds = 0
        final_acc = 0
        for t in range(self.max_steps):

            # action = self.epsilon_greedy(state, epsilon_current)
            action = self.choose_action(state)
            next_state, reward, done, acc = self.step(
                action)  # ++ during training, pick a client for next communication round
            print("episode_ct:", episode_ct, "step:", t, "acc:", acc, "action:", action, "reward:", reward, "done:",
                  done)
            print()
            total_reward += reward
            com_rounds += 1
            final_acc = acc

            self.memorize(state, action, reward, next_state, done)
            self.replay()  # sample a mini-batch from the replay buffer to train the DQN model
            state = next_state

            if done:
                break

            if t % self.target_update == 0:
                self.update_target_model()

        return total_reward, com_rounds, final_acc

    def replay(self):

        if len(self.memory) > self.batch_size:
            print("Replaying...")
            sample_batch = random.sample(self.memory, self.batch_size)
            states = []
            target_q = []
            for state, action, reward, next_state, done in sample_batch:
                states.append(state)
                # need to use the model to predict the q values
                q = self.dqn_model.predict([state], verbose=0)[0]
                # print("rest")

                # then update the experiencd action value using the target model while keeping the other action values the same
                if done:
                    q[action] = reward
                else:
                    q[action] = reward + self.gamma * np.max(self.target_model.predict([next_state], verbose=0)[0])

                target_q.append(q)

            states = np.array(states)
            target_q = np.array(target_q)

            print("Fit dqn_model")
            self.dqn_model.fit(states, target_q, epochs=1, verbose=0)
            print("Replay done.")

    # Federated learning phases
    def dqn_selection(self, action):

        sample_clients_list = [self.clients[action]]

        return sample_clients_list

    def calculate_reward(self, accuracy_this_round):

        target_accuracy = self.config.target_accuracy
        xi = self.config.reward_xi  # in article set to 64
        reward = xi ** (accuracy_this_round - target_accuracy) - 1

        return reward

    def calculate_reward_difference(self, cur_acc):

        prev_acc = self.prev_accuracy
        print("prev_acc:", prev_acc)
        print("cur_acc:", cur_acc)
        xi = self.config.reward_xi
        if cur_acc >= prev_acc:
            reward = xi ** (cur_acc - prev_acc)  # positive rewards based on improvement
        else:
            reward = - xi ** (prev_acc - cur_acc)  # negative rewards if testing acc drops

        return reward

    def step(self, action):

        accuracy, next_state = self.dqn_round(random=False, action=action)

        # calculate the reward based on the accuracy and the number of communication rounds
        if self.config.reward_fun == "target":
            reward = self.calculate_reward(accuracy)
        elif self.config.reward_fun == "difference":
            reward = self.calculate_reward_difference(accuracy)

        # rest the prev_accuracy
        self.prev_accuracy = accuracy

        # determine if the episode is done based on if reaching the target testing accuracy
        if accuracy >= self.config.target_accuracy:
            done = True
        else:
            done = False

        return next_state, reward, done, accuracy

    def profiling(self, clients, train_dqn=False):


        for client in self.clients:
            client.step()

        clients_weights_pca = self.pca.transform(np.array([flatten_weights(self.clients_weights)]))
        t_start = time.time()
        print("Start building the PCA transformer...")
        # self.pca = PCA(n_components=self.pca_n_components)
        self.pca = PCA(n_components=3)

        # dump clients_weights_pca out to pkl file for plotting
        clients_weights_pca_fn = 'output/clients_weights_pca.pkl'
        pk.dump(clients_weights_pca, open(clients_weights_pca_fn, "wb"))
        print("clients_weights_pca dumped to", clients_weights_pca_fn)

        # # dump clients_prefs
        # clients_prefs_fn = 'output/clients_prefs.pkl'
        # pk.dump(clients_prefs, open(clients_prefs_fn, "wb"))
        # print("clients_prefs dumped to", clients_prefs_fn)

        print("Built PCA transformer, time: {:.2f} s".format(time.time() - t_start))

        # save pca model out to pickl file
        pca_model_fn = "/model/pca_model.pkl"
        pk.dump(self.pca, open(pca_model_fn, "wb"))
        print("PCA model dumped to", pca_model_fn)

        for learner_id, learner in enumerate(self.global_learners_ensemble):
            learners = [client.learners_ensemble[learner_id] for client in self.clients]
            average_learners(learners, learner, weights=self.clients_weights)

        self.update_clients()
        server_weights_pca = self.pca.transform(np.array([flatten_weights(self.global_learners_ensemble[0].model)]))

        print("shape of server_weights_pca: ", server_weights_pca.shape)

        return clients_weights_pca, server_weights_pca

    """
    def getPCAWeight(self,weight):
        weight_flatten_array = self.flatten_weights(weight)
       ## demision = int(math.sqrt(weight_flatten_array.size))
        # weight_flatten_array = np.abs(weight_flatten_array)
        # sorted_array = np.sort(weight_flatten_array)
        # reverse_array = sorted_array[::-1]

        demision = weight_flatten_array.size
        weight_flatten_matrix = np.reshape(weight_flatten_array,(10,int(demision/10)))

        pca = PCA(n_components=10)
        pca.fit_transform(weight_flatten_matrix)
        newWeight = pca.transform(weight_flatten_matrix)
        # newWeight = reverse_array[0:100]

        return  newWeight
    """

    # Server operations
    def profile_all_clients(self, train_dqn):

        # all clients send updated weights to server, the server will do FedAvg
        # And then run  PCA and store the transformed weights

        print("Start profiling all clients...")

        # Perform profiling on all clients
        clients_weights_pca, server_weights_pca = self.profiling()

        # save the initial pca weights for each client + server
        self.pca_weights_clientserver_init = np.vstack((clients_weights_pca, server_weights_pca))
        print("shape of self.pca_weights_clientserver_init: ", self.pca_weights_clientserver_init.shape)

        # save a copy for later update in DQN training episodes
        self.pca_weights_clientserver = self.pca_weights_clientserver_init.copy()

        print('self.pca_weights_clientserver.shape:', self.pca_weights_clientserver.shape)

    def save_model(self, model, path):
        path += '/global_' + "case"
        torch.save(model.state_dict(), path)
        # logging.info('Saved global model: {}'.format(path))

    def load_model(self):
        model_path = self.config.paths_model
        # Set up global model
        self.model = DQNModule()
        self.save_model(self.model, model_path)
        print("Saved initial global model.")

