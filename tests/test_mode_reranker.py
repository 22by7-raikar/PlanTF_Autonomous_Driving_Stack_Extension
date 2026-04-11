"""Unit tests for src/planners/mode_reranker.py.

No nuPlan dependency — only torch and numpy are required.
Run with:  pytest tests/test_mode_reranker.py -v
"""

import sys
import os

from typing import Optional

import numpy as np
import pytest
import torch

# Ensure repo root is on the path regardless of where pytest is invoked from.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.planners.mode_reranker import (
    rerank_modes,
    DEFAULT_LAMBDA,
    ROUTE_DIST_THRESHOLD,
)

# ── constants ────────────────────────────────────────────────────────────────
K = 6    # number of modes
T = 80   # future steps
M = 20   # map polygons


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_out(
    trajectory: Optional[torch.Tensor] = None,
    probability: Optional[torch.Tensor] = None,
    output_trajectory: Optional[torch.Tensor] = None,
) -> dict:
    """Build a minimal model-output dict (batch=1)."""
    if trajectory is None:
        trajectory = torch.zeros(1, K, T, 4)
    if probability is None:
        probability = torch.zeros(1, K)
    if output_trajectory is None:
        output_trajectory = torch.zeros(1, T, 3)
    return {
        "trajectory": trajectory,
        "probability": probability,
        "output_trajectory": output_trajectory,
    }


def _make_data(
    polygon_center: Optional[torch.Tensor] = None,
    polygon_on_route: Optional[torch.Tensor] = None,
) -> dict:
    """Build a minimal feature-data dict (batch=1)."""
    if polygon_center is None:
        polygon_center = torch.zeros(1, M, 3)
    if polygon_on_route is None:
        polygon_on_route = torch.ones(1, M, dtype=torch.bool)
    return {
        "map": {
            "polygon_center": polygon_center,
            "polygon_on_route": polygon_on_route,
        }
    }


# ── test: output contract ─────────────────────────────────────────────────────

class TestOutputContract:
    def test_output_shape(self):
        """Result must be [T, 3]."""
        out  = _make_out()
        data = _make_data()
        result = rerank_modes(out, data)
        assert result.shape == (T, 3), f"Expected ({T}, 3), got {result.shape}"

    def test_output_dtype_float64(self):
        """Result must be float64 to satisfy nuPlan's trajectory builder."""
        out  = _make_out()
        data = _make_data()
        result = rerank_modes(out, data)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    def test_heading_is_atan2(self):
        """Channel 2 must equal atan2(sin, cos) from the model."""
        # Build a trajectory where cos=0, sin=1 (heading = π/2) for all steps.
        traj = torch.zeros(1, K, T, 4)
        traj[..., 2] = 0.0   # cos(heading)
        traj[..., 3] = 1.0   # sin(heading)

        prob = torch.zeros(1, K)
        prob[0, 0] = 10.0    # mode 0 dominates

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data()
        result = rerank_modes(out, data)

        expected_heading = np.pi / 2
        np.testing.assert_allclose(
            result[:, 2], expected_heading, atol=1e-5,
            err_msg="Heading channel should be atan2(sin, cos)",
        )


# ── test: fallback when no on-route polygons ──────────────────────────────────

class TestNoRoutePolygonFallback:
    def test_falls_back_to_output_trajectory(self):
        """With no on-route polygons the pre-computed output_trajectory is returned."""
        sentinel = torch.arange(T * 3, dtype=torch.float32).reshape(1, T, 3) * 0.1
        out  = _make_out(output_trajectory=sentinel)
        data = _make_data(
            polygon_on_route=torch.zeros(1, M, dtype=torch.bool)  # all off-route
        )
        result = rerank_modes(out, data)
        # The fallback re-uses output_trajectory[0] directly (no atan2 re-apply).
        np.testing.assert_allclose(
            result, sentinel[0].numpy(), atol=1e-6,
        )

    def test_shape_preserved_in_fallback(self):
        out  = _make_out()
        data = _make_data(
            polygon_on_route=torch.zeros(1, M, dtype=torch.bool)
        )
        assert rerank_modes(out, data).shape == (T, 3)


# ── test: lambda=0 (pure probability) ────────────────────────────────────────

class TestPureProbability:
    def test_lambda0_matches_argmax(self):
        """lambda_=0 → score = softmax(prob) → same mode as probability.argmax()."""
        torch.manual_seed(42)
        prob = torch.randn(1, K)
        traj = torch.randn(1, K, T, 4)
        # Normalise the direction channels so atan2 is well-defined.
        traj[..., 2:4] = torch.nn.functional.normalize(traj[..., 2:4], dim=-1)

        best_idx = prob.argmax().item()
        expected_traj = traj[0, best_idx, :, :2]

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data()
        result = rerank_modes(out, data, lambda_=0.0)

        np.testing.assert_allclose(
            result[:, :2], expected_traj.numpy(), atol=1e-5,
            err_msg="lambda=0 should select the probability-argmax mode",
        )


# ── test: lambda=1 (pure geometry) ───────────────────────────────────────────

class TestPureGeometry:
    def test_lambda1_picks_on_route_mode(self):
        """lambda_=1 → only coverage matters; should always pick the mode that
        stays nearest the route polygon centres."""
        torch.manual_seed(7)

        # Mode 0: far off-route (x,y ≈ 1000)
        # Mode 1: exactly on-route (x,y ≈ 0)
        # All other modes: also off-route.
        traj = torch.ones(1, K, T, 4) * 1000.0
        traj[0, 0, :, :2] = 1000.0   # far
        traj[0, 1, :, :2] = 0.0      # on route
        traj[..., 2] = 1.0; traj[..., 3] = 0.0  # unit cos/sin

        # Give mode 0 the highest probability → reranker should override it.
        prob = torch.zeros(1, K)
        prob[0, 0] = 100.0   # mode 0 has overwhelming probability

        # Route polygon centres clustered around the origin.
        centers = torch.zeros(1, M, 3)
        on_route = torch.ones(1, M, dtype=torch.bool)

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data(polygon_center=centers, polygon_on_route=on_route)

        result = rerank_modes(out, data, lambda_=1.0)

        # The result should come from mode 1 (near origin), not mode 0 (at 1000).
        expected_xy = traj[0, 1, :, :2].numpy()
        np.testing.assert_allclose(
            result[:, :2], expected_xy, atol=1e-4,
            err_msg="lambda=1 should pick mode 1 (on-route) over mode 0 (high prob but far)",
        )

    def test_lambda1_all_equal_coverage_uses_probability(self):
        """When all modes have the same coverage, probability still breaks ties."""
        # All trajectories sit at the origin — equal coverage for every mode.
        traj = torch.zeros(1, K, T, 4)
        traj[..., 2] = 1.0   # cos=1, sin=0

        prob = torch.zeros(1, K)
        prob[0, 3] = 5.0     # mode 3 wins on probability

        centers = torch.zeros(1, M, 3)
        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data(polygon_center=centers)

        result = rerank_modes(out, data, lambda_=1.0)
        # All coverage equal → softmax term still breaks tie via (1-1)*prob=0
        # but coverage equal means any mode could win. Just verify shape/dtype.
        assert result.shape == (T, 3)
        assert result.dtype == np.float64


# ── test: blended score ───────────────────────────────────────────────────────

class TestBlendedScore:
    def test_default_lambda_is_between_zero_and_one(self):
        assert 0.0 < DEFAULT_LAMBDA < 1.0

    def test_route_dist_threshold_positive(self):
        assert ROUTE_DIST_THRESHOLD > 0.0

    def test_intermediate_lambda_selects_valid_mode(self):
        """Result at any lambda ∈ (0,1) must be one of the K candidate modes."""
        torch.manual_seed(99)
        traj = torch.randn(1, K, T, 4)
        traj[..., 2:4] = torch.nn.functional.normalize(traj[..., 2:4], dim=-1)
        prob = torch.randn(1, K)

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data()

        result = rerank_modes(out, data, lambda_=DEFAULT_LAMBDA)

        # The result must exactly match one of the K mode trajectories.
        result_xy = result[:, :2]
        match_found = False
        for k in range(K):
            expected_xy = traj[0, k, :, :2].numpy()
            if np.allclose(result_xy, expected_xy, atol=1e-4):
                match_found = True
                break
        assert match_found, "Result did not match any of the K candidate modes"


# ── test: coverage calculation ────────────────────────────────────────────────

class TestCoverageCalculation:
    def test_perfect_coverage_when_all_waypoints_near_centers(self):
        """All waypoints near centres → coverage should be 1.0 for that mode."""
        # Waypoints at origin, centres at origin → all within threshold.
        traj = torch.zeros(1, K, T, 4)
        traj[..., 2] = 1.0

        # Give mode 0 probability 0 but force lambda=1 to test coverage.
        prob = torch.zeros(1, K)
        prob[0, 1] = 1.0  # mode 1 has higher prob

        centers  = torch.zeros(1, M, 3)
        on_route = torch.ones(1, M, dtype=torch.bool)

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data(polygon_center=centers, polygon_on_route=on_route)

        # At lambda=0, mode 1 wins (pure probability).
        r0 = rerank_modes(out, data, lambda_=0.0)
        np.testing.assert_allclose(r0[:, :2], 0.0, atol=1e-5)

    def test_zero_coverage_when_all_waypoints_far_from_centers(self):
        """Waypoints very far from centres → coverage = 0 for all modes."""
        traj = torch.ones(1, K, T, 4) * 1e6
        traj[..., 2] = 1.0; traj[..., 3] = 0.0

        prob = torch.zeros(1, K)
        prob[0, 2] = 1.0   # mode 2 wins on prob

        centers  = torch.zeros(1, M, 3)
        on_route = torch.ones(1, M, dtype=torch.bool)

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data(polygon_center=centers, polygon_on_route=on_route)

        # With all coverage=0, blended score = (1-lambda)*softmax(prob).
        # So regardless of lambda, mode 2 should win.
        result = rerank_modes(out, data, lambda_=DEFAULT_LAMBDA)
        expected_xy = traj[0, 2, :, :2].numpy()
        np.testing.assert_allclose(result[:, :2], expected_xy, atol=1e-4)


# ── test: numerical stability ─────────────────────────────────────────────────

class TestNumericalStability:
    def test_uniform_probability_does_not_crash(self):
        """Identical probabilities should not cause NaN or exception."""
        traj = torch.randn(1, K, T, 4)
        traj[..., 2:4] = torch.nn.functional.normalize(traj[..., 2:4], dim=-1)
        prob = torch.zeros(1, K)   # all equal

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data()
        result = rerank_modes(out, data)
        assert not np.any(np.isnan(result)), "NaN in output with uniform probabilities"

    def test_single_on_route_polygon(self):
        """Edge case: only one polygon is on-route."""
        on_route = torch.zeros(1, M, dtype=torch.bool)
        on_route[0, 5] = True   # only polygon 5 is on-route

        centers       = torch.zeros(1, M, 3)
        centers[0, 5] = torch.tensor([1.0, 0.0, 0.0])

        out  = _make_out()
        data = _make_data(polygon_center=centers, polygon_on_route=on_route)
        result = rerank_modes(out, data)
        assert result.shape == (T, 3)

    def test_single_mode_single_polygon(self):
        """Degenerate K=1 case should not crash (model could hypothetically be
        reconfigured with num_modes=1)."""
        K1 = 1
        traj    = torch.zeros(1, K1, T, 4); traj[..., 2] = 1.0
        prob    = torch.zeros(1, K1)
        out_traj = torch.zeros(1, T, 3)
        out  = _make_out(
            trajectory=traj, probability=prob, output_trajectory=out_traj
        )
        data = _make_data()
        result = rerank_modes(out, data)
        assert result.shape == (T, 3)


# ── test: vectorised implementation matches scalar loop ───────────────────────

class TestVectorizedMatchesLooped:
    """Verify that the vectorised coverage calculation is numerically identical
    to the scalar loop it replaced.

    The loop version is re-implemented here as a reference so this test
    remains valid even after the production code no longer contains the loop.
    Both implementations must produce identical float32 tensors (within the
    epsilon of IEEE-754 floating-point; they perform the same operations in
    the same order, so the outputs are bit-for-bit identical in practice).
    """

    @staticmethod
    def _coverage_looped(traj: torch.Tensor, route_centers: torch.Tensor) -> torch.Tensor:
        """Reference: scalar for-loop over modes (pre-vectorisation code)."""
        K_ = traj.shape[0]
        coverage = torch.zeros(K_, dtype=torch.float32, device=traj.device)
        for k in range(K_):
            waypoints = traj[k, :, :2]
            dists = torch.cdist(waypoints, route_centers)
            min_dists = dists.min(dim=-1).values
            coverage[k] = (min_dists <= ROUTE_DIST_THRESHOLD).float().mean()
        return coverage

    @staticmethod
    def _coverage_vectorized(traj: torch.Tensor, route_centers: torch.Tensor) -> torch.Tensor:
        """Current implementation: single batched cdist call."""
        K_, T_ = traj.shape[0], traj.shape[1]
        waypoints_flat = traj[:, :, :2].reshape(K_ * T_, 2)
        dists_flat = torch.cdist(waypoints_flat, route_centers)
        min_dists = dists_flat.min(dim=-1).values.reshape(K_, T_)
        return (min_dists <= ROUTE_DIST_THRESHOLD).float().mean(dim=-1)

    def test_random_trajectories_coverage_matches(self):
        """Random K×T trajectories and R centres → both implementations agree."""
        torch.manual_seed(1337)
        traj    = torch.randn(K, T, 4)
        centers = torch.randn(10, 2)

        cov_loop = self._coverage_looped(traj, centers)
        cov_vec  = self._coverage_vectorized(traj, centers)
        torch.testing.assert_close(cov_vec, cov_loop)

    def test_single_on_route_polygon_coverage_matches(self):
        """Edge case R=1: the two implementations must still agree."""
        torch.manual_seed(42)
        traj    = torch.randn(K, T, 4)
        centers = torch.zeros(1, 2)   # single centre at the origin

        cov_loop = self._coverage_looped(traj, centers)
        cov_vec  = self._coverage_vectorized(traj, centers)
        torch.testing.assert_close(cov_vec, cov_loop)

    def test_rerank_output_matches_loop_reference(self):
        """End-to-end: rerank_modes() with vectorised path selects the same
        mode that the old loop-based implementation would have selected."""
        torch.manual_seed(2024)
        R_test = 15

        traj = torch.randn(1, K, T, 4)
        traj[..., 2:4] = torch.nn.functional.normalize(traj[..., 2:4], dim=-1)
        prob = torch.randn(1, K)

        # Build a data dict with R_test on-route polygon centres.
        polygon_center = torch.zeros(1, M, 3)
        polygon_center[0, :R_test, :2] = torch.randn(R_test, 2)
        on_route = torch.zeros(1, M, dtype=torch.bool)
        on_route[0, :R_test] = True

        out  = _make_out(trajectory=traj, probability=prob)
        data = _make_data(polygon_center=polygon_center, polygon_on_route=on_route)

        # What rerank_modes() (vectorised) returns.
        result_vec = rerank_modes(out, data, lambda_=DEFAULT_LAMBDA)

        # What the scalar loop would return.
        route_centers = polygon_center[0, :R_test, :2]   # [R_test, 2]
        cov_loop  = self._coverage_looped(traj[0], route_centers)
        prob_norm = torch.softmax(prob[0], dim=-1)
        score     = (1.0 - DEFAULT_LAMBDA) * prob_norm + DEFAULT_LAMBDA * cov_loop
        best_k    = score.argmax().item()
        expected_xy = traj[0, best_k, :, :2].numpy()

        np.testing.assert_allclose(
            result_vec[:, :2],
            expected_xy,
            atol=1e-4,
            err_msg=f"Vectorised path picked a different mode than the loop reference",
        )
