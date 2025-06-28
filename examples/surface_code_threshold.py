import stim
import sinter
import matplotlib.pyplot as plt

from beliefmatching import BeliefMatchingSinterDecoder


# Generates surface code circuit tasks
def generate_example_tasks():
    for p in [0.006, 0.007, 0.008, 0.009, 0.01]:
        for d in [3, 5, 7]:
            yield sinter.Task(
                circuit=stim.Circuit.generated(
                    rounds=d,
                    distance=d,
                    after_clifford_depolarization=p,
                    after_reset_flip_probability=p,
                    before_measure_flip_probability=p,
                    before_round_data_depolarization=p,
                    code_task="surface_code:rotated_memory_x",
                ),
                json_metadata={
                    "p": p,
                    "d": d,
                },
            )


def main():
    # Collect the samples for beliefmatching
    samples = sinter.collect(
        num_workers=4,
        max_shots=1_000_000,
        max_errors=1000,
        tasks=generate_example_tasks(),
        decoders=["beliefmatching"],
        custom_decoders={"beliefmatching": BeliefMatchingSinterDecoder()},
        print_progress=True,
    )

    # Also collect samples for pymatching, for comparison. Since pymatching is much faster we will
    # collect more shots.
    samples += sinter.collect(
        num_workers=4,
        max_shots=10_000_000,
        max_errors=10_000,
        tasks=generate_example_tasks(),
        decoders=["pymatching"],
        print_progress=True,
    )
    # Plot the data
    fig, ax = plt.subplots(1, 1)
    sinter.plot_error_rate(
        ax=ax,
        stats=samples,
        group_func=lambda stat: f"{stat.decoder}, d={stat.json_metadata['d']}",
        x_func=lambda stat: stat.json_metadata["p"],
    )
    ax.loglog()
    ax.grid()
    ax.set_title("Logical Error Rate vs Physical Error Rate")
    ax.set_ylabel("Logical Error Probability (per shot)")
    ax.set_xlabel("Physical Error Rate")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
