# BeliefMatching

An implementation of the [belief-matching](https://arxiv.org/abs/2203.04948) decoder, using 
[pymatching](https://github.com/oscarhiggott/PyMatching) for the minimum-weight perfect matching (MWPM) subroutine and 
the [ldpc](https://pypi.org/project/ldpc/) library for the belief propagation (BP) subroutine.
Belief-matching is more accurate than the MWPM decoder alone when hyperedge error mechanisms are present in the error 
model.
Belief matching algorithm has the same worst-case complexity as minimum-weight perfect matching, and the average 
expected complexity is roughly linear in the size of the decoding problem (Tanner graph).

However, note that this particular implementation is much (>100x) slower than just using the pymatching (v2) 
decoder alone, since it has not been optimised for performance.
For example, for each shot, belief propagation is run on the full Tanner graph (stim `DetectorErrorModel`) with 
the output used to construct a new instance of a pymatching `Matching` object.
A much more performant implementation could be written by more tightly integrating the BP and MWPM subroutines.
Note that this implementation also uses the [ldpc](https://pypi.org/project/ldpc/) library for BP, which uses a 
parallel BP schedule, and does not support the serial BP schedule shown to have slightly improved accuracy 
for belief-matching in the appendix of [this paper](https://arxiv.org/abs/2207.06431).

## Installation

To install beliefmatching, run:
```shell
pip install -e .
```
from the root directory.

## Usage

Here is an example of how the decoder can be used directly with Stim:

```python
import stim
import numpy as np
from beliefmatching import BeliefMatching

num_shots = 100
d = 5
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
dem = circuit.detector_error_model(decompose_errors=True)

sampler = circuit.compile_detector_sampler()
shots, observables = sampler.sample(num_shots, separate_observables=True)

bm = BeliefMatching(dem, max_bp_iters=20)

num_mistakes = 0

for i in range(shots.shape[0]):
    predicted_observables = bm.decode(shots[i, :])
    num_mistakes += not np.array_equal(predicted_observables, observables[i, :])

print(f"{num_mistakes}/{num_shots}")  # prints 4/100
```

### Sinter integration

To integrate with [sinter](https://github.com/quantumlib/Stim/tree/main/glue/sample), you can use the 
`sinter.BeliefMatchingSinterDecoder` class, which inherits from `sinter.Decoder`.
To use it, you can use the `custom_decoders` argument when using `sinter.collect`:

```python
import sinter
from beliefmatching import BeliefMatchingSinterDecoder

samples = sinter.collect(
    num_workers=4,
    max_shots=1_000_000,
    max_errors=1000,
    tasks=generate_example_tasks(),
    decoders=['beliefmatching'],
    custom_decoders={'beliefmatching': BeliefMatchingSinterDecoder()}
)
```

A complete example using sinter (including the definition of `generate_example_tasks` and plotting) can be found in the 
`examples/surface_code_threshold.py` file.

Note that this sinter integration uses `sinter.Decoder` which, as of 8th December 2022, is only available in in the 
latest pre-release distributions on PyPI. Therefore the sinter dependency is set as `sinter>=1.11.dev1670280005` in 
the `setup.py`.

## Tests

Tests can be run by installing pytest with 
```shell
pip install pytest
```

and running 
```shell
pytest tests
```

