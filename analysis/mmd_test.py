import numpy as np
from typing import Callable, List
import math

from utils.types import MMDResult


def hamming_kernel(s1: str, s2: str) -> float:
    """
    Computes the Hamming kernel between two strings.
    The Hamming kernel is defined as the number of matching character positions.

    Args:
        s1: First string.
        s2: Second string.

    Returns:
        The count of matching character positions.
    """
    return float(sum(1 for c1, c2 in zip(s1, s2) if c1 == c2))


def compute_mmd(
    samples_a: List[str], samples_b: List[str], kernel: Callable[[str, str], float]
) -> float:
    """
    Computes the Maximum Mean Discrepancy (MMD) squared statistic between two sets of samples.

    Formula: MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]

    Args:
        samples_a: First list of string samples.
        samples_b: Second list of string samples.
        kernel: Kernel function that takes two strings and returns a float.

    Returns:
        The MMD squared statistic.
    """
    n = len(samples_a)
    m = len(samples_b)

    if n == 0 or m == 0:
        return 0.0

    # Compute kernel matrices
    K_XX = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K_XX[i, j] = kernel(samples_a[i], samples_a[j])

    K_YY = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            K_YY[i, j] = kernel(samples_b[i], samples_b[j])

    K_XY = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            K_XY[i, j] = kernel(samples_a[i], samples_b[j])

    term1 = np.mean(K_XX)
    term2 = np.mean(K_YY)
    term3 = np.mean(K_XY)

    return float(term1 + term2 - 2 * term3)


def permutation_test(
    samples_a: List[str],
    samples_b: List[str],
    observed_mmd: float,
    n_permutations: int,
    kernel: Callable[[str, str], float] = hamming_kernel,
) -> float:
    """
    Performs a permutation test to compute the p-value of the observed MMD statistic
    under the null hypothesis H0: P = Q.

    Args:
        samples_a: First list of string samples.
        samples_b: Second list of string samples.
        observed_mmd: The MMD statistic computed on the original samples.
        n_permutations: Number of permutations to perform.
        kernel: Kernel function to use (defaults to hamming_kernel).

    Returns:
        The estimated p-value.
    """
    n = len(samples_a)
    m = len(samples_b)

    if n == 0 or m == 0:
        return 1.0

    pooled = samples_a + samples_b
    total_samples = n + m

    # Precompute the full kernel matrix for the pooled samples to speed up permutations
    K_pooled = np.zeros((total_samples, total_samples))
    for i in range(total_samples):
        for j in range(i, total_samples):
            val = kernel(pooled[i], pooled[j])
            K_pooled[i, j] = val
            K_pooled[j, i] = val

    exceed_count = 0

    for _ in range(n_permutations):
        # Randomly permute indices
        indices = np.random.permutation(total_samples)
        idx_a = indices[:n]
        idx_b = indices[n:]

        # Compute MMD for this permutation using the precomputed kernel matrix
        term1 = np.mean(K_pooled[np.ix_(idx_a, idx_a)])
        term2 = np.mean(K_pooled[np.ix_(idx_b, idx_b)])
        term3 = np.mean(K_pooled[np.ix_(idx_a, idx_b)])

        perm_mmd = term1 + term2 - 2 * term3

        if perm_mmd >= observed_mmd:
            exceed_count += 1

    return float(exceed_count / n_permutations)


def mmd_test(
    samples_a: List[str],
    samples_b: List[str],
    kernel: str = "hamming",
    n_permutations: int = 1000,
    alpha: float = 0.05,
) -> MMDResult:
    """
    Performs the MMD two-sample test to compare two distributions of strings.

    Args:
        samples_a: First list of string samples.
        samples_b: Second list of string samples.
        kernel: Name of the kernel to use (currently supports "hamming").
        n_permutations: Number of permutations for the p-value computation.
        alpha: Significance level for rejecting the null hypothesis.

    Returns:
        MMDResult containing the test statistic, p-value, and test decision.
    """
    n = len(samples_a)
    m = len(samples_b)

    if n == 0 or m == 0:
        return MMDResult(
            mmd_statistic=0.0,
            p_value=1.0,
            reject_null=False,
            n_samples_a=n,
            n_samples_b=m,
            n_permutations=n_permutations,
            effect_size=0.0,
        )

    if kernel == "hamming":
        kernel_fn = hamming_kernel
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    observed_mmd = compute_mmd(samples_a, samples_b, kernel_fn)

    p_value = permutation_test(
        samples_a=samples_a,
        samples_b=samples_b,
        observed_mmd=observed_mmd,
        n_permutations=n_permutations,
        kernel=kernel_fn,
    )

    reject_null = p_value < alpha
    effect_size = float(math.sqrt(max(0.0, observed_mmd)))

    return MMDResult(
        mmd_statistic=observed_mmd,
        p_value=p_value,
        reject_null=reject_null,
        n_samples_a=n,
        n_samples_b=m,
        n_permutations=n_permutations,
        effect_size=effect_size,
    )
