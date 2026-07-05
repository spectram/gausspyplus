"""pytest tests for the training stage (gradient descent for the AGD smoothing parameters)."""
import numpy as np

from gausspyplus.training import gradient_descent


def _make_training_data(n_spectra=3, n_channels=200):
    """Synthetic training set: one clean Gaussian component per spectrum."""
    x_values = np.arange(n_channels, dtype=float)
    rng = np.random.default_rng(0)
    amp, mean, fwhm = 1.0, 100.0, 12.0
    stddev = fwhm / gradient_descent.CONVERSION_STD_TO_FWHM
    data_list, means, fwhms, amps = [], [], [], []
    for _ in range(n_spectra):
        spectrum = amp * np.exp(-((x_values - mean) ** 2) / (2 * stddev**2))
        data_list.append(spectrum + rng.normal(0, 0.05, n_channels))
        means.append([mean])
        fwhms.append([fwhm])
        amps.append([amp])
    return {
        "data_list": data_list,
        "error": [0.05] * n_spectra,
        "x_values": x_values,
        "means": means,
        "fwhms": fwhms,
        "amplitudes": amps,
    }


def test_single_training_example_recovers_component():
    """Run the real AGD decomposition on one synthetic spectrum (no multiprocessing)."""
    training_data = _make_training_data()
    kwargs = {
        "j": 0,
        "vel": training_data["x_values"],
        "data": training_data["data_list"],
        "errors": [np.ones(len(training_data["x_values"])) * err for err in training_data["error"]],
        "alpha1": 2.0,
        "alpha2": 6.0,
        "plot": False,
        "verbose": False,
        "SNR_thresh": 3.0,
        "SNR2_thresh": 3.0,
        "phase": "two",
        "means": training_data["means"],
        "FWHMs": training_data["fwhms"],
        "amps": training_data["amplitudes"],
    }
    n_correct, n_guess, n_true = gradient_descent.single_training_example(kwargs)
    assert n_true == 1
    assert n_correct == 1


def test_train_convergence_logic():
    """Exercise the gradient-descent loop and convergence bookkeeping with a cheap analytic objective."""

    def fake_objective(alpha1, alpha2, training_data, **kwargs):
        return -np.log(1.0 / (1.0 + (alpha1 - 2.5) ** 2 + 0.5 * (alpha2 - 5.0) ** 2))

    alpha1, alpha2, gd = gradient_descent.train(
        objective_function=fake_objective,
        training_data=_make_training_data(),
        alpha1_initial=2.0,
        alpha2_initial=6.0,
        iterations=60,
        learning_rate=0.5,
        window_size=3,
        iterations_for_convergence=2,
        phase="two",
        verbose=False,
    )
    assert np.isfinite(alpha1) and np.isfinite(alpha2)
    assert abs(alpha1 - 2.5) < 1.0
