import os
from typing import List

import pymatching
from scipy.sparse import csc_matrix
import stim
import numpy as np
import sinter

from beliefmatching import iter_set_xor, dict_to_csc_matrix, detector_error_model_to_check_matrices, BeliefMatching, \
    BeliefMatchingSinterDecoder

test_dir = os.path.dirname(os.path.realpath(__file__))


def assert_csc_eq(sparse_mat: csc_matrix, dense_mat: List[List[int]]) -> None:
    assert (sparse_mat != csc_matrix(dense_mat)).nnz == 0


def test_iter_set_xor():
    assert iter_set_xor([[0, 1, 2, 5]]) == frozenset((0, 1, 2, 5))
    assert iter_set_xor([[0, 1], [1, 2]]) == frozenset((0, 2))
    assert iter_set_xor([[4, 1, 9, 2], [4, 2, 5, 10]]) == frozenset((1, 5, 9, 10))


def test_dict_to_csc_matrix():
    m = dict_to_csc_matrix(
        {
            0: frozenset((0, 3)),
            1: frozenset((2, 4)),
            3: frozenset((1, 3))
        },
        shape=(5, 5)
    )
    assert (m != csc_matrix([[1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0],
                             [1, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0]])).nnz == 0


def test_dem_to_check_matrices():
    dem = stim.DetectorErrorModel.from_file(os.path.join(test_dir, "one_hyperedge_decomposed.dem"))
    mats = detector_error_model_to_check_matrices(dem)
    assert_csc_eq(mats.check_matrix, [[1, 1, 0],
                                      [1, 0, 1],
                                      [1, 0, 1],
                                      [1, 1, 0]])
    assert_csc_eq(mats.observables_matrix, [[1, 1, 0],
                                            [0, 1, 1]])
    assert_csc_eq(mats.edge_check_matrix, [[1, 0],
                                           [0, 1],
                                           [0, 1],
                                           [1, 0]])
    assert_csc_eq(mats.edge_observables_matrix, [[1, 0],
                                                 [1, 1]])
    assert_csc_eq(mats.hyperedge_to_edge_matrix, [[1, 1, 0], [1, 0, 1]])
    assert np.allclose(mats.priors, np.array([0.22, 0.2, 0.46]))


def generate_shot_data():
    d = 7
    p = 0.007
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        rounds=d,
        distance=d,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
        after_clifford_depolarization=p
    )
    circuit.to_file(f"surface_code_rotated_memory_x_d_{d}_p_{p}.stim")
    dem = circuit.detector_error_model(decompose_errors=True)
    dem.to_file(f"surface_code_rotated_memory_x_d_{d}_p_{p}.dem")

    num_shots = 500

    sampler = circuit.compile_detector_sampler()
    shot_data = sampler.sample(num_shots, separate_observables=False, append_observables=True)
    stim.write_shot_data_file(data=shot_data, path=f"surface_code_rotated_memory_x_d_{d}_p_{p}_{num_shots}_shots.b8",
                              num_detectors=dem.num_detectors, num_observables=dem.num_observables, format="b8")


def test_belief_matching_surface_code():
    circuit = stim.Circuit.from_file(os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.stim"))
    dem = stim.DetectorErrorModel.from_file(os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007.dem"))

    bm_dem_init = BeliefMatching(dem, max_bp_iters=20, bp_method="product_sum")
    bm_dem_class = BeliefMatching.from_detector_error_model(dem, max_bp_iters=20, bp_method="product_sum")
    bm_circuit_init = BeliefMatching(circuit, max_bp_iters=20, bp_method="product_sum")
    bm_circuit_class = BeliefMatching.from_stim_circuit(circuit, max_bp_iters=20, bp_method="product_sum")

    matching = pymatching.Matching.from_detector_error_model(dem)
    shot_data = stim.read_shot_data_file(
        path=os.path.join(test_dir, "surface_code_rotated_memory_x_d_7_p_0.007_500_shots.b8"), format="b8",
        num_detectors=dem.num_detectors, num_observables=dem.num_observables)

    shots = shot_data[:, 0:dem.num_detectors]
    observables = shot_data[:, dem.num_detectors:]

    predicted_observables = matching.decode_batch(shots)
    num_mistakes_pm = np.sum(np.any(predicted_observables != observables, axis=1))
    assert num_mistakes_pm == 11

    for bm in (bm_dem_init, bm_dem_class):
        predicted_observables = bm.decode_batch(shots)
        num_mistakes_bm = np.sum(np.any(predicted_observables != observables, axis=1))
        assert num_mistakes_bm == 7

    for bm in (bm_circuit_init, bm_circuit_class):
        num_mistakes_bm = 0
        for i in range(shots.shape[0]):
            predicted_obs_bm = bm.decode(shots[i, :])
            actual_obs = observables[i, :]
            num_mistakes_bm += not np.array_equal(actual_obs, predicted_obs_bm)
        assert num_mistakes_bm == 7


def generate_trivial_circuit_task():
        yield sinter.Task(
            circuit=stim.Circuit("""X_ERROR(0.1) 0
M 0
DETECTOR rec[-1]
OBSERVABLE_INCLUDE(4) rec[-1]"""),
        )


def test_belief_matching_sinter_multiple_obs():
    sinter.collect(
        num_workers=1,
        max_shots=1_000_000,
        max_errors=1000,
        tasks=generate_trivial_circuit_task(),
        decoders=['beliefmatching'],
        custom_decoders={'beliefmatching': BeliefMatchingSinterDecoder()}
    )
