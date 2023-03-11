import time

import numpy as np
from multiprocessing import Pool, cpu_count
import math
from numpy import linalg as LA
from utils.sadmm_updates import z_update, u_update, stochastic_x_update
from utils.sadmm_utils import get_value, calculate_accuracy
import time as t


class NetworkLassoRunner:

    def __init__(self, G):
        self.G = G

    def run_sadmm(self, lamb, rho, c, max_iterations, num_features, datasets, datasets_test, x, z, u, z_residuals,
                  u_residuals, logger):
        pool = Pool(1)

        num_edges = len(self.G.edges)
        num_nodes = len(self.G.nodes)

        eabs = math.pow(10, -2)
        erel = math.pow(10, -3)
        (r, s, epri, edual, counter) = (1, 1, 0, 0, 0)
        A = np.zeros((2 * num_edges, num_nodes))
        (sqn, sqp) = (math.sqrt(num_nodes * num_features), math.sqrt(2 * num_features * num_edges))
        for edge in sorted(self.G.edges):
            A[counter * 2, edge[0]] = 1
            A[counter * 2 + 1, edge[1]] = 1
            counter += 1

        elapsed_iterations = 0
        calculate_accuracy(x, num_nodes, datasets_test, logger, elapsed_iterations)
        start_time = t.time()
        while elapsed_iterations < max_iterations and (r > epri or s > edual or elapsed_iterations < 1):
            # start_time = t.time()
            mu = 1 / float(elapsed_iterations + 1)

            x_data = []
            for node_id in self.G.nodes:
                neighs = self.G[node_id]
                local_dataset = datasets[node_id]
                neighbour_data = np.zeros((2 * len(neighs), num_features))
                neigh_counter = 0
                for neighbour_id in neighs:
                    z_val = get_value(z, node_id, neighbour_id)
                    u_val = get_value(u, node_id, neighbour_id)
                    neighbour_data[neigh_counter * 2, :] = z_val
                    neighbour_data[neigh_counter * 2 + 1, :] = u_val
                    neigh_counter += 1
                x_data.append((node_id, rho, c, local_dataset, neighbour_data, x[:, node_id], mu))
            new_x = pool.map(stochastic_x_update, x_data)
            x = np.array(new_x).reshape(num_nodes, num_features).T

            edge_data = []
            for edge in sorted(self.G.edges):
                n1 = edge[0]
                n2 = edge[1]
                x1 = x[:, n1].T
                x2 = x[:, n2].T
                u1 = get_value(u, n1, n2)
                u2 = get_value(u, n2, n1)
                local_edge_data = (lamb, rho, x1, x2, u1, u2, self.G.get_edge_data(n1, n2)['weight'])
                edge_data.append(local_edge_data)
            new_z = pool.map(z_update, edge_data)

            ztemp = np.array(new_z).reshape(num_edges * 2, num_features).T
            s = LA.norm(rho * np.dot(A.transpose(), (ztemp - z_residuals).transpose()))  # For dual residual

            z_counter = 0
            for edge in sorted(self.G.edges):
                z[edge] = new_z[z_counter]
                z_residuals[:, z_counter * 2] = new_z[z_counter][0]
                z_residuals[:, z_counter * 2 + 1] = new_z[z_counter][1]
                z_counter += 1

            u_data = []
            for edge in sorted(self.G.edges):
                n1 = edge[0]
                n2 = edge[1]
                x1 = x[:, n1].T
                x2 = x[:, n2].T
                z1 = get_value(z, n1, n2)
                z2 = get_value(z, n2, n1)
                u1 = get_value(u, n1, n2)
                u2 = get_value(u, n2, n1)
                local_u_data = (x1, x2, z1, z2, u1, u2)
                u_data.append(local_u_data)
            new_u = pool.map(u_update, u_data)

            u_counter = 0
            for edge in sorted(self.G.edges):
                u[edge] = new_u[u_counter]
                u_residuals[:, u_counter * 2] = new_u[u_counter][0]
                u_residuals[:, u_counter * 2 + 1] = new_u[u_counter][1]
                u_counter += 1

            epri = sqp * eabs + erel * max(LA.norm(np.dot(A, x.transpose()), 'fro'), LA.norm(z_residuals, 'fro'))
            edual = sqn * eabs + erel * LA.norm(np.dot(A.transpose(), u_residuals.transpose()), 'fro')
            r = LA.norm(np.dot(A, x.transpose()) - z_residuals.transpose(), 'fro')
            s = s
            elapsed_iterations += 1
            print("Time Iteration: " + str(elapsed_iterations) + ", Time: " + str(t.time() - start_time))
            print("r: " + str(r) + ", epri: " + str(epri) + " | s: " + str(s) + ", edual: " + str(edual))
            calculate_accuracy(x, num_nodes, datasets_test, logger, elapsed_iterations)

        pool.close()
        pool.join()
        calculate_accuracy(x, num_nodes, datasets_test)
        return x, z, u, z_residuals, u_residuals, elapsed_iterations

    def run(self, args, num_features, datasets, datasets_test, c, logger):
        print("Running Stochastic Network Lasso...")

        rho = 1.0

        num_nodes = len(self.G.nodes)
        num_edges = len(self.G.edges)

        x = np.zeros((num_features, num_nodes))
        z = {}
        u = {}
        z_residuals = np.zeros((num_features, 2 * num_edges))
        u_residuals = np.zeros((num_features, 2 * num_edges))

        counter = 0
        for edge in sorted(self.G.edges):
            # Initialized z and u
            z[edge] = (np.zeros((1, num_features)), np.zeros((1, num_features)))
            u[edge] = (np.zeros((1, num_features)), np.zeros((1, num_features)))
            counter += 1

        x, z, u, z_residuals, u_residuals, iters = self.run_sadmm(args.sadmm_lambda, rho, c, args.n_rounds, num_features,
                                                                      datasets, datasets_test, x, z, u, z_residuals,
                                                                      u_residuals, logger)

        return calculate_accuracy(x, num_nodes, datasets_test)
