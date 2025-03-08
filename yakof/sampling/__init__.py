"""
Ensemble-Aware Distribution Sampling
====================================

This package provides the Sampler class for extracting Sample from a
probability.DensityOrMass. The Sampler.sample method allows one to draw
samples from a distribution, either by random variates sampling or by
exhaustive sampling, depending on the distribution support size, the
number of desired samples, and other options. The return value is a
ListOfSamples, which is a list of Sample instances. Each Sample instance
contains the sampled value, its probability, and its weight. The weight
should be used when generating ensembles. The ListOfSamples class also
provides a method to compute the total probability of the samples, which,
under the assumption that variables inside the ensemble are independent,
allows to estimate the overall probability covered by the ensemble.
"""

from dataclasses import dataclass

from typing import Generic, TypeVar, cast

from .. import probability


T = TypeVar("T")
"""Generic type of a value in a discrete distribution."""


@dataclass(frozen=True)
class Sample(Generic[T]):
    """Result of sampling from a distribution.

    Fields:
        probability: the probability of this sample.
        value: sampled value.
        weight: weight of the sample for generating ensembles.
    """

    probability: float
    value: T
    weight: float


@dataclass(frozen=True)
class ListOfSamples(Generic[T]):
    """List of samples from a distribution.

    Fields:
        samples: sequence of samples.
    """

    samples: list[Sample[T]]

    def __len__(self) -> int:
        """Returns the number of samples in this list."""
        return len(self.samples)

    def total_probability(self) -> float:
        """Computes and returns the total probability represented by
        the samples contained in this list of samples."""
        # Statistical note: dedup potentially-duplicate discrete samples
        unique = {s.value: s.probability for s in self.samples}
        return sum(unique.values())


class Sampler(Generic[T]):
    """Extracts samples from a probability.DensityOrMass.

    Args:
        dist: probability density or mass.
    """

    def __init__(self, dist: probability.DensityOrMass[T]):
        self.dist = dist

    @staticmethod
    def _validate_count(count: int) -> int:
        """Validates that the count is a positive integer."""
        if count <= 0:
            raise ValueError("count must be a positive integer")
        return count

    def sample(
        self,
        nr: int = 1,
        *,
        subset: list[T] | None = None,
        force_sample: bool = False,
    ) -> ListOfSamples[T]:
        """Samples points from the underlying distribution.

        Args:
            nr: number of samples to return if using random variates sampling.
            subset: optional subset of values to consider for sampling.
            force_sample: whether to unconditionally use random variates sampling.

        Returns:
            ListOfSamples[T]: list of samples from the distribution

        Raises:
            ValueError: in case of invalid arguments.

        Algorithm
        ---------

        The `subset` argument transforms the distribution into a discrete
        distribution by restricting the support to the subset and renormalizing
        the probabilities. For a continuous distribution, this means
        evaluating the PDF on the subset and normalizing the probabilities. For
        a discrete distribution, this means selecting the subset of values and
        normalizing the probabilities.

        Subsequent steps operate on the transformed distribution.

        We support two distinct sampling modes:

            1. Random Variates Sampling: we extract `nr` random variates from
            the transformed distribution. Each sample has a weight of 1/nr and
            a probability equal to its probability in the distribution. This
            means that the same point may appear multiple times in the output.

            2. Exhaustive Sampling: we return all the samples within the
            probability mass class. This mode is only available for discrete
            distributions. Note that a continuous distribution may become
            a discrete distribution when a subset is provided.

        We use random variates sampling when `nr` is lower than the transformed
        distribution support size (remember that continuous distributions have
        an infinite support) or when the `force_sample` flag is set.

        Otherwise, we use exhaustive sampling.
        """
        # === Validation ===
        nr = self._validate_count(nr)

        # === Subset ===
        #
        # The presence of a subset forces us to switch from a potentially
        # continuous distribution to a discrete distribution.
        dist = self._discretize(subset) if subset else self.dist

        # Defer to the _sample method - implementation note: we use a
        # separate class method to make self.dist inaccessible and thus
        # avoid potentially programming errors.
        return self._sample(dist, nr, subset, force_sample)

    @classmethod
    def _sample(
        cls,
        dist: probability.DensityOrMass[T],
        nr: int,
        subset: list[T] | None,
        force_sample: bool,
    ) -> ListOfSamples[T]:

        # === Support Size Considerations ===
        #
        # A large (possibly infinite) support size means we fall back
        # to extracting random variates from the distribution.
        force_sample = force_sample or nr < dist.support_size()

        # === Random Variates Sampling ===
        #
        # In this sampling mode, we just extract values from the
        # distribution, and return `nr` samples. If `nr` is negative,
        # the method used for sampling will raise an exception.
        if force_sample:
            return cls._rvs(dist, nr)

        # === Exhaustive Sampling ===
        #
        # If we end up here, `dist.support_size() >= count` and
        # definitely `dist.support_size() < math.inf`. So, we know
        # the distribution is discrete and we return all the
        # samples within the probability mass class.
        #
        # Statistical note: since we're going to exhaustive sampling,
        # each of the samples occurs once with weight==probability.
        assert isinstance(dist, probability.Mass)
        return cls._exhaustive_sampling(cast(probability.Mass[T], dist))

    @staticmethod
    def _exhaustive_sampling(dist: probability.Mass[T]) -> ListOfSamples[T]:
        return ListOfSamples(
            samples=[
                Sample(probability=y, value=x, weight=y)
                for x, y in dist.dist.items()
            ]
        )

    @staticmethod
    def _rvs(dist: probability.DensityOrMass[T], nr: int) -> ListOfSamples[T]:
        # Sample from the distribution
        xs = dist.sample(nr)

        # Evaluate the corresponding probabilities
        ys = dist.evaluate(xs)

        # Ensure that probs is a list of float
        ys = [float(ys)] if isinstance(ys, float | int) else ys

        # Determine the weight of each sample
        #
        # Statistical note: each sample has the same weight in the ensemble
        # and occurs as frequently in the output as it is likely.
        weight = 1.0 / nr if nr > 0 else 0.0

        # Create the list of samples
        return ListOfSamples(
            samples=[
                Sample(probability=prob, value=point, weight=weight)
                for prob, point in zip(ys, xs)
            ]
        )

    def _discretize(self, subset: list[T]) -> probability.DensityOrMass[T]:
        # Evaluate the distribution on the subset
        ys = self.dist.evaluate(subset)

        # Ensure that ys is a list of float
        ys = [float(ys)] if isinstance(ys, float | int) else ys

        # Normalize the probabilities
        tot = sum(ys)
        pairs = [(k, v / tot if tot > 0 else 0) for k, v in zip(subset, ys)]

        # Account for duplicate entries
        summed = {}
        for k, v in pairs:
            summed[k] = summed.get(k, 0) + v

        # Convert the dictionary to a mass function
        return probability.mass(summed)
