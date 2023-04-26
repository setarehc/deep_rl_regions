import numpy as np
import os
import gc
from functools import partial
from multiprocessing import Pool
import multiprocessing as mpz

#-------------------------------------------------------------------------------------------------------------------#
#---------------------------------- Implementation borrows from Hanin and Rolnick ----------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

"""
This file has all the tools needed for counting regions and transitions over a 1D trajectory.
"""

class LinearRegion1D:
    def __init__(self, param_min, param_max, fn_weight, fn_bias, next_layer_off, state=None):
        self._min = param_min
        self._max = param_max
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._next_layer_off = next_layer_off
        self.state = [] if state is None else state

    def get_new_regions(self, new_weight_n, new_bias_n, n):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        if weight_n == 0:
            min_image = bias_n
            max_image = bias_n
        elif weight_n >= 0:
            min_image = weight_n * self._min + bias_n
            max_image = weight_n * self._max + bias_n
        else:
            min_image = weight_n * self._max + bias_n
            max_image = weight_n * self._min + bias_n
        if 0 < min_image:
            return [self]
        elif 0 > max_image:
            self._next_layer_off.append(n)
            return [self]
        else:
            if weight_n == 0:
                return [self]
            else:
                preimage = (-bias_n) / weight_n
                if preimage in [self._max, self._min]:
                    return [self]
                next_layer_off0 = list(np.copy(self._next_layer_off))
                next_layer_off1 = list(np.copy(self._next_layer_off))
                if weight_n >= 0:
                    next_layer_off0.append(n)
                else:
                    next_layer_off1.append(n)
                region0 = LinearRegion1D(self._min, preimage, self._fn_weight, self._fn_bias, next_layer_off0, state=list(np.copy(self.state)))
                region1 = LinearRegion1D(preimage, self._max, self._fn_weight, self._fn_bias, next_layer_off1, state=list(np.copy(self.state)))
                return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight.T).ravel()
        self._fn_bias = (np.dot(self._fn_bias, new_weight.T) + new_bias).ravel()
        self._fn_weight[self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        
        new_state = [1 for _ in range(len(new_bias))]
        for off_idx in self._next_layer_off:
            new_state[off_idx] = 0
        self.state.extend(new_state)
        
        self._next_layer_off = []

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def next_layer_off(self):
        return self._next_layer_off

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))


def count_regions_1d(the_weights, the_biases, input_line_weight, input_line_bias,
                     param_min=-np.inf, param_max=np.inf, return_regions=False, consolidate_dead_regions=False):
    regions = [LinearRegion1D(param_min, param_max, input_line_weight, input_line_bias, [])]
    depth = len(the_weights)
    for k in range(depth):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][n, :], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    if return_regions:
        return regions
    else:
        return len(regions)


def get_wandbs(policy):
    weights = []
    biases = []
    policy_net_depth = len(policy.mlp_extractor.policy_net) // 2
    for i in range(policy_net_depth):
        weights.append(policy.mlp_extractor.policy_net[i*2].weight.cpu().detach().numpy())
        biases.append(policy.mlp_extractor.policy_net[i*2].bias.cpu().detach().numpy())
    return weights, biases


def normalize_line_segment(env, point1, point2):
    point1_scaled = env.normalize_obs(point1)
    point2_scaled = env.normalize_obs(point2)
    mid_point = (point1 + point2) / 2
    mid_point_scaled = env.normalize_obs(mid_point)
    assert all(mid_point_scaled == (point1_scaled + point2_scaled) / 2) or (abs(np.linalg.norm(mid_point_scaled - (point1_scaled + point2_scaled) / 2)) < 0.0001)
    return point1_scaled, point2_scaled


def get_visited_regions_analytic(env, policy, states):
    unique_regions = set()
    num_regions_list = []
    [the_weights, the_biases] = get_wandbs(policy)
    param_min = 0
    param_max = 1

    for i in range(len(states)-1):
        point1, point2 = (states[i], states[i+1]) if env is None else normalize_line_segment(env, states[i], states[i+1])
        input_line_weight, input_line_bias = (point2-point1, point1)
        
        regions = count_regions_1d(the_weights, the_biases, input_line_weight, input_line_bias,
                                   param_min=param_min, param_max=param_max, return_regions=True)

        unique_regions.update(set([str(region.state) for region in regions]))
        num_regions_list.append(len(regions))

    num_regions = len(unique_regions)
    num_transitions = np.array(num_regions_list).sum() - len(num_regions_list)

    return num_regions, num_transitions


def get_visited_regions_analytic(env, policy, states, normalize_obs=True): # normalize by env (ignore last parameter: no normalization in case env doesn't have any normalization)
    # TODO: input env similar to compute_trajectory_length
    unique_regions = set()
    num_regions_list = []
    [the_weights, the_biases] = get_wandbs(policy)
    param_min = 0
    param_max = 1

    for i in range(len(states)-1):
        point1, point2 = normalize_line_segment(env, states[i], states[i+1]) if normalize_obs else (states[i], states[i+1])
        input_line_weight, input_line_bias = (point2-point1, point1)
        
        regions = count_regions_1d(the_weights, the_biases, input_line_weight, input_line_bias,
                                   param_min=param_min, param_max=param_max, return_regions=True)

        unique_regions.update(set([str(region.state) for region in regions]))
        num_regions_list.append(len(regions))

    num_regions = len(unique_regions)
    num_transitions = np.array(num_regions_list).sum() - len(num_regions_list)

    return num_regions, num_transitions


def mp_helper_f(env, states, normalize_obs, the_weights, the_biases, param_min, param_max, i):
    point1, point2 = normalize_line_segment(env, states[i], states[i+1]) if normalize_obs else (states[i], states[i+1])
    input_line_weight, input_line_bias = (point2-point1, point1)

    regions = count_regions_1d(the_weights, the_biases, input_line_weight, input_line_bias,
                               param_min=param_min, param_max=param_max, return_regions=True)
    return regions


def get_visited_regions_analytic_parallel(env, policy, states, normalize_obs=True):
    all_regions = set()
    region_lengths = []
    [the_weights, the_biases] = get_wandbs(policy)
    param_min = 0
    param_max = 1

    ncpus = int(os.environ.get("SLURM_CPUS_PER_TASK", default=1))
    pool = Pool(processes=ncpus, maxtasksperchild=1000)

    all_regions = pool.map(partial(mp_helper_f, env, states, normalize_obs, the_weights, the_biases, param_min, param_max),
                                range(len(states)-1))
    pool.close()
    pool.join()
    gc.collect()
    num_transitions = 0
    all_cells = []
    for regions in all_regions:
        num_transitions += len(regions) - 1
        for region in regions:
            all_cells.append(region.state)

    regions = np.unique(all_cells, axis=0)
    num_regions = len(regions)
    
    return num_regions, num_transitions


def compute_trajectory_length(states, env=None):
    # Unnormalized
    euclidean_length = 0
    for i in range(len(states)-1):
        p2, p1 = (states[i+1], states[i]) if env is None else (env.normalize_obs(states[i+1]), env.normalize_obs(states[i]))
        euclidean_length += np.linalg.norm(p2-p1)
    return euclidean_length