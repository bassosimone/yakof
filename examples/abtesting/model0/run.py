import numpy as np


def run_with_module(module):
    # TODO(bassosimone): this is not the correct representation of the ensemble
    # and will crash the `yakof` model evaluation. What do do here?
    fake_ensemble = [
        (1, {"weekday": "monday"}),
    ]

    grid = {
        module.drink_customers: np.linspace(0, 100, 10),
        module.food_customers: np.linspace(0, 100, 10),
    }

    print("Now we can evaluate our model")
    print(module.model.evaluate(grid, fake_ensemble))


def run_with_dtmodel():
    import model_dtm

    run_with_module(model_dtm)


def run_with_yakof():
    import model_yakof

    run_with_module(model_yakof)


if __name__ == "__main__":
    run_with_dtmodel()
    run_with_yakof()
