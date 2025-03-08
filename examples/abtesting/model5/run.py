import numpy as np


from dt_model import Ensemble


def run_with_module(module):
    ensemble = Ensemble(
        module.model, {module.CV_weekday: module.days}, cv_ensemble_size=7
    )

    grid = {
        module.drink_customers: np.linspace(0, 100, 11),
        module.food_customers: np.linspace(0, 100, 11),
    }

    print("Now we can evaluate our model")
    print(module.model.evaluate(grid, ensemble))


def run_with_dtmodel():
    import model_dtm

    run_with_module(model_dtm)


def run_with_yakof():
    import model_yakof

    run_with_module(model_yakof)


if __name__ == "__main__":
    run_with_dtmodel()
    run_with_yakof()
