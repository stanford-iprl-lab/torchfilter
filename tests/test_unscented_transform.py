import hypothesis
import torch
from hypothesis import strategies as st

import diffbayes


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_identity(dim: int):
    """Check unscented transform on multivariate Gaussians.
    """
    N = 5

    # Deterministic tests are nice!
    torch.random.manual_seed(0)

    # Get input mean, covariance
    input_mean = torch.randn((N, dim))
    input_cov_root = torch.randn((N, dim, dim))
    input_cov = input_cov_root @ input_cov_root.transpose(-1, -2)

    # Perform unscented transform; we use a big spread value for numerical precision
    unscented_transform = diffbayes.utils.UnscentedTransform(dim=dim, alpha=1.0)
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_cov)
    output_mean, output_cov = unscented_transform.compute_distribution(sigma_points)

    # Check outputs
    expected_output_mean = input_mean
    expected_output_cov = input_cov
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_cov, expected_output_cov)


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_linear(dim: int):
    """Check unscented transform on linearly transformed multivariate Gaussians.
    """
    N = 5

    # Deterministic tests are nice!
    torch.random.manual_seed(0)

    # Get input mean, covariance
    input_mean = torch.randn((N, dim))
    input_cov_root = torch.randn((N, dim, dim))
    input_cov = input_cov_root @ input_cov_root.transpose(-1, -2)

    # Sample linear transformation matrix
    A = torch.randn((N, dim, dim))

    # Perform unscented transform; we use a big spread value for numerical precision
    unscented_transform = diffbayes.utils.UnscentedTransform(dim=dim, alpha=1.0)
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_cov)
    sigma_points = (A[:, None, :, :] @ sigma_points[:, :, :, None]).squeeze(-1)
    output_mean, output_cov = unscented_transform.compute_distribution(sigma_points)

    # Check outputs
    expected_output_mean = (A @ input_mean[:, :, None]).squeeze(-1)
    expected_output_cov = A @ input_cov @ A.transpose(-1, -2)
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_cov, expected_output_cov, atol=1e-4, rtol=1e-4)
