"""Cross comparison testing of the two models."""

from yakof.molveno import orig, yak

import numpy as np
import random


def test_cross_fixed_case():
    """Cross compare with a fixed input case."""

    # Load models but don't run them yet
    orig_model = orig.M_Base
    yak_model = yak.M_Base

    orig_model.reset()
    yak_model.reset()

    # Instead of random ensemble, create a fixed single case
    fixed_orig_situation = {
        orig.CV_weekday: orig.Symbol("monday"),
        orig.CV_season: orig.Symbol("high"),
        orig.CV_weather: orig.Symbol("good"),
    }

    fixed_yak_situation = {
        yak.CV_weekday: yak.Symbol("monday"),
        yak.CV_season: yak.Symbol("high"),
        yak.CV_weather: yak.Symbol("good"),
    }

    # Fixed tourist and excursionist values
    tourists = np.array([1000, 2000, 3000])
    excursionists = np.array([500, 1500, 2500])

    # Evaluate models with fixed inputs and while resetting
    # the random seed right before evaluation
    #
    # See https://xkcd.com/221/
    np.random.seed(4)
    random.seed(4)
    orig_result = orig_model.evaluate(
        {orig.PV_tourists: tourists, orig.PV_excursionists: excursionists},
        [(1.0, fixed_orig_situation)],  # Single ensemble member with weight 1.0
    )

    np.random.seed(4)
    random.seed(4)
    yak_result = yak_model.evaluate(
        {yak.PV_tourists: tourists, yak.PV_excursionists: excursionists},
        [(1.0, fixed_yak_situation)],  # Single ensemble member with weight 1.0
    )

    # Compare results with detailed output
    print("Original model result:", orig_result)
    print("Yakof model result:", yak_result)

    # Build a map between a name of a constraint and the constraint
    assert yak_model.field_elements is not None
    yak_cmap = {
        key.name: value for key, value in yak_model.field_elements.items() if key
    }

    assert orig_model.field_elements is not None
    orig_cmap = {
        key.name: value for key, value in orig_model.field_elements.items() if key
    }

    # Ensure we have the same constraint names and that they are four
    assert set(orig_cmap.keys()) == set(yak_cmap.keys())
    assert len(orig_cmap) == 4

    # Collect all differences for reporting
    failures = []

    # Proceed to check ~equality for each constraint
    for key in sorted(orig_cmap.keys()):
        orig_c = orig_cmap[key]
        yak_c = yak_cmap[key]

        # Basic shape check
        if orig_c.shape != yak_c.shape:
            failures.append(
                f"Shape mismatch for {key}: {orig_c.shape} vs {yak_c.shape}"
            )
            continue

        # Check if values are close enough
        if not np.allclose(orig_c, yak_c, rtol=1e-5, atol=1e-8):
            # Calculate differences for diagnosis
            abs_diff = np.abs(orig_c - yak_c)
            rel_diff = abs_diff / (np.abs(orig_c) + 1e-10)  # Avoid division by zero

            # Find the worst offenders (highest absolute difference)
            max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

            diff_info = (
                f"Constraint {key} has differences:\n"
                f"  Max absolute diff: {np.max(abs_diff):.6e} at index {max_diff_idx}\n"
                f"  Mean absolute diff: {np.mean(abs_diff):.6e}\n"
                f"  Max relative diff: {np.max(rel_diff):.6f}\n"
                f"  Values at max diff: orig={orig_c[max_diff_idx]:.6f}, yak={yak_c[max_diff_idx]:.6f}\n"
                f"  First few pairs of values (orig, yak):\n"
            )

            # Sample some value pairs for comparison
            flat_orig = orig_c.flatten()
            flat_yak = yak_c.flatten()
            sample_size = min(5, len(flat_orig))

            for i in range(sample_size):
                diff_info += f"    [{i}]: {flat_orig[i]:.6f}, {flat_yak[i]:.6f} (diff: {abs_diff.flatten()[i]:.6e})\n"

            failures.append(diff_info)

    # If we have any failures, report them all at once
    if failures:
        failure_message = "Model comparison failed:\n" + "\n".join(failures)
        assert False, failure_message
