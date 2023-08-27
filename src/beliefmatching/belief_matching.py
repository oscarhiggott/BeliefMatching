from typing import List, FrozenSet, Dict, Tuple, Union
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
    """
    Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` giving the indices of nonzero
    rows in each column.

    Parameters
    ----------
    elements_dict : dict[int, frozenset[int]]
        A dictionary giving the indices of nonzero rows in each column. `elements_dict[i]` is a frozenset of ints
        giving the indices of nonzero rows in column `i`.
    shape : Tuple[int, int]
        The dimensions of the matrix to be returned

    Returns
    -------
    scipy.sparse.csc_matrix
        The `scipy.sparse.csc_matrix` check matrix defined by `elements_dict` and `shape`
    """
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


def detector_error_model_to_check_matrices(
        dem: stim.DetectorErrorModel,
        allow_undecomposed_hyperedges: bool = False
) -> DemMatrices:
    """
    Convert a `stim.DetectorErrorModel` into a `DemMatrices` object.

    Parameters
    ----------
    dem : stim.DetectorErrorModel
        A stim DetectorErrorModel
    allow_undecomposed_hyperedges: bool
        If True, don't raise an exception if a hyperedge is not decomposable. Instead, the hyperedge `h` is still added
        to the `DemMatrices.check_matrix`, `DemMatrices.observables_matrix` and `DemMatrices.priors` but it will not
        have any edges in its decomposition in `DemMatrices.hyperedge_to_edge_matrix[:, h]`.
    Returns
    -------
    DemMatrices
        A collection of matrices representing the stim DetectorErrorModel
    """
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

            if len(e_dets) > 2:
                if not allow_undecomposed_hyperedges:
                    raise ValueError("A hyperedge error mechanism was found that was not decomposed into edges. "
                                     "This can happen if you do not set `decompose_errors=True` as required when "
                                     "calling `circuit.detector_error_model`.")
                else:
                    continue

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
    def __init__(
            self,
            model: Union[stim.Circuit, stim.DetectorErrorModel],
            max_bp_iters: int = 20,
            bp_method: str = "product_sum",
            **kwargs
    ):
        """
        Construct a BeliefMatching object from a `stim.Circuit` or `stim.DetectorErrorModel`

        Parameters
        ----------
        model : stim.Circuit or stim.DetectorErrorModel
            A stim.Circuit or a stim.DetectorErrorModel. If a stim.Circuit is provided, it will be converted
            into a stim.DetectorErrorModel using `stim.Circuit.detector_error_model(decompose_errors=True)`.
            If a `stim.DetectorErrorModel` is provided it is important that its hyperedges are decomposed
            into edges (using `decompose_errors=True`) for BeliefMatching to provide improved accuracy over
            a standard (faster) MWPM decoder.
        max_bp_iters : int
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `max_iter` argument. Default 20
        bp_method : str
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log" and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "product_sum"
        kwargs
            Additional keyword arguments are passed to `ldpc.bp_decoder`
        """
        if isinstance(model, stim.Circuit):
            model = model.detector_error_model(decompose_errors=True)
        self._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            **kwargs
        )

    def _initialise_from_detector_error_model(
            self,
            model: stim.DetectorErrorModel,
            *,
            max_bp_iters: int = 20,
            bp_method: str = "product_sum",
            **kwargs
            ):
        self._model = model
        self._matrices = detector_error_model_to_check_matrices(self._model)
        self._bpd = bp_decoder(
            self._matrices.check_matrix,
            max_iter=max_bp_iters,
            bp_method=bp_method,
            channel_probs=self._matrices.priors,
            input_vector_type="syndrome",
            **kwargs
        )

    @classmethod
    def from_detector_error_model(
            cls,
            model: stim.DetectorErrorModel,
            *,
            max_bp_iters: int = 20,
            bp_method: str = "product_sum",
            **kwargs
    ) -> "BeliefMatching":
        """
        Construct a BeliefMatching object from a `stim.DetectorErrorModel`

        Parameters
        ----------
        model : stim.DetectorErrorModel
            A `stim.DetectorErrorModel`. It is important that the hyperedges are already decomposed
            into edges (using `decompose_errors=True`) for BeliefMatching to provide improved accuracy over
            a standard (faster) MWPM decoder.
        max_bp_iters : int
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `max_iter` argument. Default 20
        bp_method : str
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log" and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "product_sum"
        kwargs
            Additional keyword arguments are passed to `ldpc.bp_decoder`


        Returns
        -------
        BeliefMatching
            The BeliefMatching object for decoding using `model`
        """
        bm = cls.__new__(cls)
        bm._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            **kwargs
        )
        return bm

    @classmethod
    def from_stim_circuit(
            cls,
            circuit: stim.Circuit,
            *,
            max_bp_iters: int = 20,
            bp_method: str = "product_sum",
            **kwargs
    ) -> "BeliefMatching":
        """
        Construct a BeliefMatching object from a `stim.Circuit`

        Parameters
        ----------
        circuit : stim.Circuit
            A stim.Circuit. The circuit will be converted into a stim.DetectorErrorModel using
            `stim.Circuit.detector_error_model(decompose_errors=True)`.
        max_bp_iters : int
            The maximum number of interations of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `max_iter` argument. Default 20
        bp_method : str
            The method of belief-propagation to use. Passed to
            `ldpc.bp_decoder` as the `bp_method` argument. Options include "product_sum",
             "minimum_sum", "product_sum_log" and "minimum_sum_log" (see https://github.com/quantumgizmos/ldpc
             for details). Default is "product_sum"
        kwargs
            Additional keyword arguments are passed to `ldpc.bp_decoder`


        Returns
        -------
        BeliefMatching
            The BeliefMatching object for decoding using `model`
        """
        bm = cls.__new__(cls)
        model = circuit.detector_error_model(decompose_errors=True)
        bm._initialise_from_detector_error_model(
            model=model,
            max_bp_iters=max_bp_iters,
            bp_method=bp_method,
            **kwargs
        )
        return bm

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were flipped

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were flipped.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was flipped and 0 otherwise.
        """
        corr = self._bpd.decode(syndrome)
        if self._bpd.converge:
            return (self._matrices.observables_matrix @ corr) % 2
        llrs = self._bpd.log_prob_ratios
        ps_h = 1 / (1 + np.exp(llrs))
        ps_e = self._matrices.hyperedge_to_edge_matrix @ ps_h
        eps = 1e-14
        ps_e[ps_e > 1 - eps] = 1 - eps
        ps_e[ps_e < eps] = eps
        matching = pymatching.Matching.from_check_matrix(
            self._matrices.edge_check_matrix,
            weights=-np.log(ps_e),
            faults_matrix=self._matrices.edge_observables_matrix,
            use_virtual_boundary_node=True
        )
        return matching.decode(syndrome)

    def decode_batch(self, shots: np.ndarray) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BeliefMatching.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """
        predictions = np.zeros((shots.shape[0], self._matrices.observables_matrix.shape[0]), dtype=bool)
        for i in range(shots.shape[0]):
            predictions[i, :] = self.decode(shots[i, :])
        return predictions
