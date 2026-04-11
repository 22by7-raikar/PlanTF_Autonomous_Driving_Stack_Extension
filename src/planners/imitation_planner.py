import time
import contextlib
from typing import List, Optional, Type
import os
import csv

import numpy as np
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import (
    DetectionsTracks,
    Observation,
)
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner,
    PlannerInitialization,
    PlannerInput,
    PlannerReport,
)
from nuplan.planning.simulation.planner.planner_report import MLPlannerReport
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
    InterpolatedTrajectory,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper

from src.feature_builders.common.utils import rotate_round_z_axis

from .mode_reranker import rerank_modes
from .planner_utils import global_trajectory_to_states, load_checkpoint



class ImitationPlanner(AbstractPlanner):
    """
    Long-term IL-based trajectory planner, with short-term RL-based trajectory tracker.
    """

    requires_scenario: bool = False

    def __init__(
        self,
        planner: TorchModuleWrapper,
        planner_ckpt: str = None,
        replan_interval: int = 1,
        use_gpu: bool = True,
        record_latency: bool = False,
        use_nvtx: bool = False,
    ) -> None:
        """
        Initializes the ML planner class.
        :param model: Model to use for inference.
        :param record_latency: If True, append per-step latency to
            benchmarks/mini_v1/latency.csv.  Default False (off in eval runs).
        :param use_nvtx: If True, wrap each pipeline stage in an NVTX range.
            Only effective when running under `nsys profile` with a CUDA device.
            Safe to leave False in all normal evaluation runs.
        """
        if use_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        self._planner = planner
        self._planner_feature_builder = planner.get_list_of_required_feature()[0]
        self._planner_ckpt = planner_ckpt
        self._initialization: Optional[PlannerInitialization] = None

        self._future_horizon = 8.0
        self._step_interval = 0.1

        self._replan_interval = replan_interval
        self._last_plan_elapsed_step = replan_interval  # force plan at first step
        self._global_trajectory = None
        self._start_time = None

        # NVTX profiling flag.  When True, each sub-stage of _planning() is
        # wrapped in a named NVTX range visible in Nsight Systems.  The flag
        # only takes effect when device==cuda; on CPU the push/pop calls are
        # harmless no-ops but add a tiny amount of Python overhead, so it is
        # safest to only set this during deliberate profiling runs.
        self._use_nvtx = use_nvtx and (self.device.type == "cuda")

        # Runtime stats for the MLPlannerReport
        self._feature_building_runtimes: List[float] = []
        self._inference_runtimes: List[float] = []

        # Latency CSV (opt-in only — leave off for normal eval runs)
        self._record_latency = record_latency
        if self._record_latency:
            self._benchmark_dir = os.path.join(os.getcwd(), "benchmarks", "mini_v1")
            self._latency_csv = os.path.join(self._benchmark_dir, "latency.csv")
            os.makedirs(self._benchmark_dir, exist_ok=True)

            if not os.path.exists(self._latency_csv):
                with open(self._latency_csv, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "step",
                        "feature_build_ms",
                        "forward_ms",
                        "total_planner_ms",
                        "gpu_mem_mb",
                    ])

        self._step_idx = 0

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        torch.set_grad_enabled(False)

        if self._planner_ckpt is not None:
            self._planner.load_state_dict(load_checkpoint(self._planner_ckpt))

        self._planner.eval()
        self._planner = self._planner.to(self.device)
        self._initialization = initialization

        # just to trigger numba compile, no actually meaning
        rotate_round_z_axis(np.zeros((1, 2), dtype=np.float64), float(0.0))

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    # ------------------------------------------------------------------
    # NVTX helper
    # ------------------------------------------------------------------
    # A tiny context manager so every call site reads
    #   with self._nvtx("stage_name"):
    # instead of repeating the if-guard.

    @contextlib.contextmanager
    def _nvtx(self, name: str):
        if self._use_nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if self._use_nvtx:
                torch.cuda.nvtx.range_pop()

    def _planning(self, current_input: PlannerInput):
        planner_start = time.perf_counter()

        # ── Stage 1: feature building (CPU-side nuPlan preprocessing) ─────
        # This is where the nuPlan SimulationHistory is read, agents and map
        # polygons are extracted, normalised into ego-frame, and packaged as
        # tensors.  It is 100% CPU work.  In Nsight Systems you will see this
        # as dead GPU time — the GPU is idle while Python runs this code.
        with self._nvtx("feature_build"):
            feature_start = time.perf_counter()
            planner_feature = self._planner_feature_builder.get_features_from_simulation(
                current_input, self._initialization
            )
            feature_end = time.perf_counter()

        # ── Stage 2: host-to-device transfer ─────────────────────────────
        # .to_feature_tensor() converts numpy arrays to CPU torch tensors.
        # .to_device(self.device) performs the actual H2D copy (cudaMemcpy
        # under the hood).  With pageable (non-pinned) host memory, this
        # copy must wait for the GPU to finish any outstanding work first,
        # which can introduce a synchronization bubble.
        with self._nvtx("h2d_transfer"):
            planner_feature_torch = planner_feature.collate(
                [planner_feature.to_feature_tensor().to_device(self.device)]
            )

        feature_build_ms = (feature_end - feature_start) * 1000.0
        self._feature_building_runtimes.append(feature_end - feature_start)

        # ── Stage 3: model forward pass (GPU compute) ─────────────────────
        # This is the TRT/PyTorch GPU work.  The synchronize() calls around
        # it ensure the CPU waits for the GPU to finish before we record the
        # end time — without them, forward_ms would measure only the kernel
        # launch latency, not the actual execution time.
        with self._nvtx("model_forward"):
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_start = time.perf_counter()

            out = self._planner.forward(planner_feature_torch.data)

            if self.device.type == "cuda":
                torch.cuda.synchronize()
            forward_end = time.perf_counter()

        forward_ms = (forward_end - forward_start) * 1000.0

        # ── Stage 4: mode reranking (post-processing) ─────────────────────
        # rerank_modes() selects the best of the 6 candidate trajectories
        # by blending model probability with on-route coverage.  It currently
        # runs as Python + torch.cdist in a for-loop (Phase 3 will fix this).
        # On a GPU path this stage pulls data back to CPU and re-dispatches
        # small torch ops, which shows up in Nsight as fragmented kernel
        # launches after the main inference block.
        with self._nvtx("rerank_modes"):
            local_trajectory = rerank_modes(out, planner_feature_torch.data)

        planner_end = time.perf_counter()
        total_planner_ms = (planner_end - planner_start) * 1000.0

        gpu_mem_mb = 0.0
        if self.device.type == "cuda":
            gpu_mem_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)

        if self._record_latency:
            with open(self._latency_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    self._step_idx,
                    feature_build_ms,
                    forward_ms,
                    total_planner_ms,
                    gpu_mem_mb,
                ])

        self._step_idx += 1
        self._start_time = planner_start

        return local_trajectory.astype(np.float64)

    def compute_planner_trajectory(
        self, current_input: PlannerInput
    ) -> AbstractTrajectory:
        ego_state = current_input.history.ego_states[-1]    

        if self._last_plan_elapsed_step >= self._replan_interval:
            local_trajectory = self._planning(current_input)
            self._global_trajectory = self._get_global_trajectory(
                local_trajectory, ego_state
            )
            self._last_plan_elapsed_step = 0
        else:
            self._global_trajectory = self._global_trajectory[1:]

        trajectory = InterpolatedTrajectory(
            trajectory=global_trajectory_to_states(
                global_trajectory=self._global_trajectory,
                ego_history=current_input.history.ego_states,
                future_horizon=len(self._global_trajectory) * self._step_interval,
                step_interval=self._step_interval,
            )
        )

        self._inference_runtimes.append(time.perf_counter() - self._start_time)
        self._last_plan_elapsed_step += 1

        return trajectory

    def generate_planner_report(self, clear_stats: bool = True) -> PlannerReport:
        """Inherited, see superclass."""
        report = MLPlannerReport(
            compute_trajectory_runtimes=self._compute_trajectory_runtimes,
            feature_building_runtimes=self._feature_building_runtimes,
            inference_runtimes=self._inference_runtimes,
        )
        if clear_stats:
            self._compute_trajectory_runtimes: List[float] = []
            self._feature_building_runtimes = []
            self._inference_runtimes = []

        return report

    def _get_global_trajectory(self, local_trajectory: np.ndarray, ego_state: EgoState):
        origin = ego_state.rear_axle.array
        angle = ego_state.rear_axle.heading

        global_position = (
            rotate_round_z_axis(np.ascontiguousarray(local_trajectory[..., :2]), -angle)
            + origin
        )
        global_heading = local_trajectory[..., 2] + angle

        global_trajectory = np.concatenate(
            [global_position, global_heading[..., None]], axis=1
        )

        return global_trajectory
