from typing import Tuple

import hypothesis
import torch
from hypothesis import strategies as st

import torchfilter


def _gen_test_data(dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    N = 1

    # Deterministic tests are nice!
    torch.random.manual_seed(0)

    # Get input mean, covariance
    input_mean = torch.randn((N, dim))
    input_covariance_root = torch.randn((N, dim, dim))
    input_covariance = input_covariance_root @ input_covariance_root.transpose(-1, -2)
    return input_mean, input_covariance


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_julier_identity(dim: int):
    """Check unscented transform on multivariate Gaussians.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(dim=dim)
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_covariance)
    output_mean, output_covariance = unscented_transform.compute_distribution(
        sigma_points
    )

    # Check outputs
    expected_output_mean = input_mean
    expected_output_covariance = input_covariance
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_covariance, expected_output_covariance)


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_julier_linear(dim: int):
    """Check unscented transform on linearly transformed multivariate Gaussians.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)

    # Sample linear transformation matrix
    A = torch.randn(input_covariance.shape)

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(dim=dim)
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_covariance)
    sigma_points = (A[:, None, :, :] @ sigma_points[:, :, :, None]).squeeze(-1)
    output_mean, output_covariance = unscented_transform.compute_distribution(
        sigma_points
    )

    # Check outputs
    expected_output_mean = (A @ input_mean[:, :, None]).squeeze(-1)
    expected_output_covariance = A @ input_covariance @ A.transpose(-1, -2)
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(
        output_covariance, expected_output_covariance, atol=1e-4, rtol=1e-4
    )


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_merwe_identity(dim: int):
    """Check unscented transform on multivariate Gaussians.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(
        dim=dim,
        sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=2e-1),
    )
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_covariance)
    output_mean, output_covariance = unscented_transform.compute_distribution(
        sigma_points
    )

    # Check outputs
    expected_output_mean = input_mean
    expected_output_covariance = input_covariance
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_covariance, expected_output_covariance)


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_merwe_linear(dim: int):
    """Check unscented transform on linearly transformed multivariate Gaussians.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)

    # Sample linear transformation matrix
    A = torch.randn(input_covariance.shape)

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(
        dim=dim,
        sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=2e-1),
    )
    sigma_points = unscented_transform.select_sigma_points(input_mean, input_covariance)
    sigma_points = (A[:, None, :, :] @ sigma_points[:, :, :, None]).squeeze(-1)
    output_mean, output_covariance = unscented_transform.compute_distribution(
        sigma_points
    )

    # Check outputs
    expected_output_mean = (A @ input_mean[:, :, None]).squeeze(-1)
    expected_output_covariance = A @ input_covariance @ A.transpose(-1, -2)
    torch.testing.assert_allclose(
        output_mean, expected_output_mean, atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_allclose(
        output_covariance, expected_output_covariance, atol=1e-4, rtol=1e-4
    )


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_merwe_square_root(dim: int):
    """Check unscented transform square root formulation on multivariate Gaussians.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(
        dim=dim,
        sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=2e-1),
    )
    sigma_points = unscented_transform.select_sigma_points_square_root(
        input_mean, torch.cholesky(input_covariance)
    )
    (
        output_mean,
        output_scale_tril,
    ) = unscented_transform.compute_distribution_square_root(sigma_points)

    # Check outputs
    output_covariance = output_scale_tril @ output_scale_tril.transpose(-1, -2)
    expected_output_mean = input_mean
    expected_output_covariance = input_covariance
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_covariance, expected_output_covariance)


@hypothesis.given(dim=st.integers(min_value=1, max_value=20))
def test_unscented_transform_merwe_square_root_additive_nosie(dim: int):
    """Check unscented transform square root formulation on multivariate Gaussians, with
    additive noise.
    """
    input_mean, input_covariance = _gen_test_data(dim=dim)
    N, _dim = input_mean.shape
    assert _dim == dim

    additive_noise_scale_tril = torch.tril(torch.randn((N, dim, dim)))
    additive_noise_covariance = (
        additive_noise_scale_tril @ additive_noise_scale_tril.transpose(-1, -2)
    )

    # Perform unscented transform
    unscented_transform = torchfilter.utils.UnscentedTransform(
        dim=dim,
        sigma_point_strategy=torchfilter.utils.MerweSigmaPointStrategy(alpha=2e-1),
    )
    sigma_points = unscented_transform.select_sigma_points_square_root(
        input_mean, torch.cholesky(input_covariance)
    )
    (
        output_mean,
        output_scale_tril,
    ) = unscented_transform.compute_distribution_square_root(
        sigma_points, additive_noise_scale_tril=additive_noise_scale_tril,
    )

    # Check outputs
    output_covariance = output_scale_tril @ output_scale_tril.transpose(-1, -2)
    expected_output_mean = input_mean
    expected_output_covariance = input_covariance + additive_noise_covariance
    torch.testing.assert_allclose(output_mean, expected_output_mean)
    torch.testing.assert_allclose(output_covariance, expected_output_covariance)
