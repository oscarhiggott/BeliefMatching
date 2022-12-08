from typing import List, FrozenSet, Dict, Tuple
from dataclasses import dataclass

from ldpc import bp_decoder
from scipy.sparse import csc_matrix
import numpy as np

import stim
import pymatching


def iter_set_xor(set_list: List[List[int]]) -> FrozenSet[int]:
    out = set()
    for x in set_list:
        s = set(x)
        out = (out - s) | (s - out)
    return frozenset(out)


def dict_to_csc_matrix(elements_dict: Dict[int, FrozenSet[int]], shape: Tuple[int, int]) -> csc_matrix:
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)


@dataclass
class DemMatrices:
    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    edge_check_matrix: csc_matrix
    edge_observables_matrix: csc_matrix
    hyperedge_to_edge_matrix: csc_matrix
    priors: np.ndarray


def detector_error_model_to_check_matrices(dem: stim.DetectorErrorModel) -> DemMatrices:
    hyperedge_ids: Dict[FrozenSet[int], int] = {}
    edge_ids: Dict[FrozenSet[int], int] = {}
    hyperedge_obs_map: Dict[int, FrozenSet[int]] = {}
    edge_obs_map: Dict[int, FrozenSet[int]] = {}
    priors_dict: Dict[int, float] = {}
    hyperedge_to_edge: Dict[int, FrozenSet[int]] = {}

    def handle_error(prob: float, detectors: List[List[int]], observables: List[List[int]]) -> None:
        hyperedge_dets = iter_set_xor(detectors)
        hyperedge_obs = iter_set_xor(observables)

        if hyperedge_dets not in hyperedge_ids:
            hyperedge_ids[hyperedge_dets] = len(hyperedge_ids)
            priors_dict[hyperedge_ids[hyperedge_dets]] = 0.0
        hid = hyperedge_ids[hyperedge_dets]
        hyperedge_obs_map[hid] = hyperedge_obs
        priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])

        eids = []
        for i in range(len(detectors)):
            e_dets = frozenset(detectors[i])
            e_obs = frozenset(observables[i])

            if e_dets not in edge_ids:
                edge_ids[e_dets] = len(edge_ids)
            eid = edge_ids[e_dets]
            eids.append(eid)
            edge_obs_map[eid] = e_obs

        if hid not in hyperedge_to_edge:
            hyperedge_to_edge[hid] = frozenset(eids)

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[List[int]] = [[]]
            frames: List[List[int]] = [[]]
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets[-1].append(t.val)
                elif t.is_logical_observable_id():
                    frames[-1].append(t.val)
                elif t.is_separator():
                    dets.append([])
                    frames.append([])
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix({v: k for k, v in hyperedge_ids.items()},
                                      shape=(dem.num_detectors, len(hyperedge_ids)))
    observables_matrix = dict_to_csc_matrix(hyperedge_obs_map, shape=(dem.num_observables, len(hyperedge_ids)))
    priors = np.zeros(len(hyperedge_ids))
    for i, p in priors_dict.items():
        priors[i] = p
    hyperedge_to_edge_matrix = dict_to_csc_matrix(hyperedge_to_edge, shape=(len(edge_ids), len(hyperedge_ids)))
    edge_check_matrix = dict_to_csc_matrix({v: k for k, v in edge_ids.items()},
                                           shape=(dem.num_detectors, len(edge_ids)))
    edge_observables_matrix = dict_to_csc_matrix(edge_obs_map, shape=(dem.num_observables, len(edge_ids)))
    return DemMatrices(
        check_matrix=check_matrix,
        observables_matrix=observables_matrix,
        edge_check_matrix=edge_check_matrix,
        edge_observables_matrix=edge_observables_matrix,
        hyperedge_to_edge_matrix=hyperedge_to_edge_matrix,
        priors=priors
    )


class BeliefMatching:
    def __init__(self, model: stim.DetectorErrorModel, max_bp_iters: int = 20):
        self.model = model
        self.matrices = detector_error_model_to_check_matrices(model)
        self.bpd = bp_decoder(
            self.matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method="product_sum",
            channel_probs=self.matrices.priors
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        corr = self.bpd.decode(syndrome)
        if self.bpd.converge:
            return (self.matrices.observables_matrix @ corr) % 2
        llrs = self.bpd.log_prob_ratios
        ps_h = 1 / (1 + np.exp(llrs))
        ps_e = self.matrices.hyperedge_to_edge_matrix @ ps_h
        eps = 1e-14
        ps_e[ps_e > 1 - eps] = 1 - eps
        ps_e[ps_e < eps] = eps
        matching = pymatching.Matching.from_check_matrix(
            self.matrices.edge_check_matrix,
            weights=-np.log(ps_e),
            faults_matrix=self.matrices.edge_observables_matrix,
            use_virtual_boundary_node=True
        )
        return matching.decode(syndrome)
