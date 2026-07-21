#!/usr/bin/env python3
"""
il_dagger.py  —  DAgger Rollout and Expert Relabeling  (Phase 4)

Provides:
  - PolicyProvider: abstract/pluggable policy inference backend
    (disabled, python_module, onnx).
  - PolicyOutput dataclass.
  - DaggerController: beta-scheduled action selection, safety override,
    expert relabeling, data recording coordination.

DAgger labels always come from the finite-observation expert at the
learner-visited state.  Learner policy NEVER sees global ESDF, precise
guide, or expert action.
"""

from __future__ import print_function, division

import math, os, time, hashlib, json, collections
import numpy as np
from dataclasses import dataclass, field

import rospy


# ============================================================================
#  PolicyOutput
# ============================================================================

@dataclass
class PolicyOutput:
    """Output of a single policy inference step."""
    valid: bool = False

    velocity_flu: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64))
    yaw_rate: float = 0.0

    inference_ms: float = 0.0

    trend_azimuth_bin: int = -1
    trend_elevation_bin: int = -1
    trend_distance_norm: float = 0.0

    hidden_state_updated: bool = False

    rejection_reason: str = ""


# ============================================================================
#  PolicyProvider  (abstract backend selector)
# ============================================================================

class PolicyProvider:
    """Pluggable policy inference backend.

    Supports:
      - "disabled": always returns invalid (pure expert mode).
      - "python_module": lazily imports a user Python class.
      - "onnx": lazily loads an ONNX model via onnxruntime.
    """

    def __init__(self, config):
        dagger_cfg = config.get("global", {}).get("dagger", {})
        policy_cfg = dagger_cfg.get("policy", {})

        self._backend = str(policy_cfg.get("backend", "disabled"))
        self._model_path = str(policy_cfg.get("model_path", ""))
        self._device = str(policy_cfg.get("device", "cpu"))
        self._recurrent = bool(policy_cfg.get("recurrent", True))
        self._inference_timeout_ms = float(
            policy_cfg.get("inference_timeout_ms", 30.0))

        self._python_module = str(policy_cfg.get("python_module", ""))
        self._python_class = str(policy_cfg.get("python_class", ""))

        self._onnx_inputs = policy_cfg.get("onnx_input_names", {})
        self._onnx_outputs = policy_cfg.get("onnx_output_names", {})

        self._model = None
        self._session = None
        self._hidden_state = None
        self._loaded = False

        self.total_inferences = 0
        self.total_inference_ms = 0.0

    @property
    def backend(self):
        return self._backend

    @property
    def loaded(self):
        return self._loaded

    def reset(self):
        """Reset hidden state for new episode."""
        self._hidden_state = None

    def load(self):
        """Lazy-load the policy model. Called once before first inference."""
        if self._loaded:
            return True

        if self._backend == "disabled":
            self._loaded = True
            return True

        if self._backend == "python_module":
            if not self._python_module or not self._python_class:
                rospy.logerr("[Policy] python_module/class not configured.")
                return False
            try:
                import importlib
                mod = importlib.import_module(self._python_module)
                cls = getattr(mod, self._python_class)
                self._model = cls()
                self._loaded = True
                rospy.loginfo("[Policy] Loaded Python module: %s.%s",
                              self._python_module, self._python_class)
            except Exception as e:
                rospy.logerr("[Policy] Failed to load Python module: %s", e)
                return False

        if self._backend == "onnx":
            if not self._model_path:
                rospy.logerr("[Policy] ONNX model_path not configured.")
                return False
            try:
                import onnxruntime as ort
                providers = ['CPUExecutionProvider']
                if self._device == 'cuda':
                    providers.insert(0, 'CUDAExecutionProvider')
                self._session = ort.InferenceSession(
                    self._model_path, providers=providers)
                self._loaded = True
                rospy.loginfo("[Policy] Loaded ONNX model: %s", self._model_path)
            except ImportError:
                rospy.logerr("[Policy] onnxruntime not installed. Install: pip install onnxruntime")
                return False
            except Exception as e:
                rospy.logerr("[Policy] Failed to load ONNX model: %s", e)
                return False

        return self._loaded

    def infer(self, depth, global_guide, vehicle_state, timestamp_s):
        """Run policy inference.

        Args:
            depth: (H, W) float32 depth image in metres.
            global_guide: dict with direction_flu, distance_norm fields.
            vehicle_state: dict with vel_flu, yaw, yaw_rate fields.
            timestamp_s: episode time.

        Returns:
            PolicyOutput with inference result.
        """
        output = PolicyOutput()
        self.total_inferences += 1

        if self._backend == "disabled":
            output.rejection_reason = "policy_disabled"
            return output

        t0 = time.monotonic()

        try:
            if self._backend == "python_module" and self._model is not None:
                result = self._model.infer(
                    depth, global_guide, vehicle_state, self._hidden_state)
                elapsed = (time.monotonic() - t0) * 1000.0
                self._unpack_result(output, result, elapsed)

            elif self._backend == "onnx" and self._session is not None:
                elapsed = self._infer_onnx(output, depth, global_guide,
                                            vehicle_state)
        except Exception as e:
            output.rejection_reason = "inference_exception: {}".format(str(e)[:80])
            rospy.logwarn("[Policy] Inference exception: %s", e)

        return output

    def _unpack_result(self, output, result, elapsed_ms):
        output.inference_ms = elapsed_ms
        self.total_inference_ms += elapsed_ms

        if result is None:
            output.rejection_reason = "model_returned_none"
            return

        if hasattr(result, 'valid') and not result.valid:
            output.rejection_reason = getattr(result, 'rejection_reason', 'invalid_output')
            return

        try:
            vel = getattr(result, 'velocity_flu', None)
            if vel is not None:
                output.velocity_flu = np.asarray(vel, dtype=np.float64).ravel()[:3]
            output.yaw_rate = float(getattr(result, 'yaw_rate', 0.0))
            output.trend_azimuth_bin = int(getattr(result, 'trend_azimuth_bin', -1))
            output.trend_elevation_bin = int(getattr(result, 'trend_elevation_bin', -1))
            output.trend_distance_norm = float(getattr(result, 'trend_distance_norm', 0.0))
            if hasattr(result, 'hidden_state'):
                self._hidden_state = result.hidden_state
                output.hidden_state_updated = True
            output.valid = True
        except Exception as e:
            output.rejection_reason = "unpack_error: {}".format(str(e)[:80])

    def _infer_onnx(self, output, depth, global_guide, vehicle_state):
        """Run ONNX inference."""
        t0 = time.monotonic()
        try:
            import onnxruntime as ort

            # Build input dict from configured names
            inputs = {}
            input_names = self._onnx_inputs

            depth_key = input_names.get("depth", "depth")
            guide_dir_key = input_names.get("global_dir", "global_dir")
            guide_dist_key = input_names.get("global_dist", "global_dist")
            vel_key = input_names.get("velocity", "velocity")
            yaw_key = input_names.get("yaw", "yaw")
            hidden_key = input_names.get("hidden", "hidden")

            # Prepare normalized depth
            if depth is not None:
                d = np.asarray(depth, dtype=np.float32)
                d = np.clip(d / 5.0, 0.0, 1.0)  # normalize by max range
                inputs[depth_key] = d[np.newaxis, np.newaxis, :, :]

            if global_guide is not None:
                gd = np.asarray(global_guide.get("direction_flu", [0, 0, 0]),
                                dtype=np.float32)
                inputs[guide_dir_key] = gd[np.newaxis, :]
                inputs[guide_dist_key] = np.array(
                    [[global_guide.get("distance_norm", 0.0)]], dtype=np.float32)

            if vehicle_state is not None:
                vf = np.asarray(vehicle_state.get("vel_flu", [0, 0, 0]),
                                dtype=np.float32)
                inputs[vel_key] = vf[np.newaxis, :]
                inputs[yaw_key] = np.array(
                    [[vehicle_state.get("yaw_rate", 0.0)]], dtype=np.float32)

            if self._recurrent and self._hidden_state is not None and hidden_key:
                inputs[hidden_key] = self._hidden_state

            out_names = [self._onnx_outputs.get(k, k) for k in
                         ["velocity", "yaw_rate", "hidden_out"]]

            result = self._session.run(out_names, inputs)

            output.velocity_flu = np.asarray(result[0], dtype=np.float64).ravel()[:3]
            output.yaw_rate = float(result[1].ravel()[0]) if len(result) > 1 else 0.0
            if len(result) > 2:
                self._hidden_state = result[2]
                output.hidden_state_updated = True
            output.valid = True

            elapsed = (time.monotonic() - t0) * 1000.0
            output.inference_ms = elapsed
            self.total_inference_ms += elapsed
            return elapsed

        except Exception as e:
            output.rejection_reason = "onnx_error: {}".format(str(e)[:80])
            return 0.0

    def model_hash(self):
        """Return SHA256 hash of model file, or empty string."""
        if not self._model_path or not os.path.isfile(self._model_path):
            return ""
        try:
            return hashlib.sha256(
                open(self._model_path, "rb").read()).hexdigest()
        except Exception:
            return ""


# ============================================================================
#  DAgger Controller
# ============================================================================

class DaggerController:
    """Coordinates DAgger rollout: beta scheduling, action selection, safety.

    DAgger labels always come from the finite-observation expert at the
    learner-visited state.
    """

    def __init__(self, config):
        dagger_cfg = config.get("global", {}).get("dagger", {})
        safety_cfg = dagger_cfg.get("safety", {})

        self._enabled = bool(dagger_cfg.get("enabled", False))
        self._round_id = int(dagger_cfg.get("round_id", 0))
        self._rollout_mode = str(dagger_cfg.get("rollout_mode", "expert"))
        self._seed = int(dagger_cfg.get("seed", 12345))

        # Beta schedule
        mixture = dagger_cfg.get("mixture", {})
        self._beta_mode = str(mixture.get("mode", "stochastic_switch"))
        self._beta_schedule = str(mixture.get("beta_schedule", "linear"))
        self._beta_start = float(mixture.get("beta_start", 1.0))
        self._beta_end = float(mixture.get("beta_end", 0.0))
        self._beta_decay_rounds = int(mixture.get("beta_decay_rounds", 5))
        self._sample_scope = str(mixture.get("sample_scope", "frame"))

        self._current_beta = self._compute_beta(self._round_id)

        # Safety
        self._safety_enabled = bool(safety_cfg.get("enabled", True))
        self._override_on_invalid = bool(
            safety_cfg.get("override_on_invalid_policy_output", True))
        self._override_on_timeout = bool(
            safety_cfg.get("override_on_policy_timeout", True))
        self._override_on_collision_risk = bool(
            safety_cfg.get("override_on_observed_collision_risk", True))
        self._min_obs_clearance = float(
            safety_cfg.get("minimum_observed_clearance_m", 0.35))
        self._min_ttc_s = float(safety_cfg.get("minimum_ttc_s", 0.60))
        self._fallback_command = str(safety_cfg.get("fallback_command", "hover"))

        # Output root
        agg = dagger_cfg.get("aggregation", {})
        self._output_root = str(agg.get("output_root",
                                         os.path.join(
                                             config.get("global", {}).get("output_dir", "."),
                                             "dagger")))
        self._preserve_episode_id = bool(agg.get("preserve_original_episode_id", True))

        # RNG per episode
        self._rng = None

        # Stats per episode
        self.expert_exec_count = 0
        self.learner_exec_count = 0
        self.safety_override_count = 0
        self.invalid_label_count = 0

    @property
    def enabled(self):
        return self._enabled

    @property
    def round_id(self):
        return self._round_id

    @property
    def rollout_mode(self):
        return self._rollout_mode

    @property
    def current_beta(self):
        return self._current_beta

    def _compute_beta(self, round_id):
        """Compute beta (expert probability) for given round."""
        if self._beta_schedule == "linear":
            progress = min(float(round_id) / max(self._beta_decay_rounds, 1), 1.0)
            return self._beta_start + progress * (self._beta_end - self._beta_start)
        return self._beta_start  # constant

    def reset_episode(self, episode_seed):
        """Reset for new episode. Creates deterministic RNG."""
        self._rng = np.random.RandomState(
            int(self._seed) + int(self._round_id) * 10000 + int(episode_seed))
        self.expert_exec_count = 0
        self.learner_exec_count = 0
        self.safety_override_count = 0

    def select_actor(self, expert_action_valid, learner_output):
        """Select which actor's command to execute.

        Args:
            expert_action_valid: bool, whether expert produced a valid action.
            learner_output: PolicyOutput from learner inference.

        Returns:
            (actor_name, safety_override, safety_reason, final_command_flu, final_yaw_rate)
        """
        # Select based on beta
        if self._rollout_mode == "expert" or not self._enabled:
            selected = "expert"
        elif self._rollout_mode == "dagger":
            if self._rng is not None:
                rv = self._rng.uniform()
            else:
                rv = 0.5
            selected = "expert" if rv < self._current_beta else "learner"
        else:
            selected = "expert"

        # Safety checks (only use observed information, NOT global ESDF)
        safety_override = False
        safety_reason = ""

        if selected == "learner" and self._safety_enabled:
            # Check learner output validity
            if (self._override_on_invalid and
                    (learner_output is None or not learner_output.valid)):
                safety_override = True
                safety_reason = "learner_output_invalid"
            elif (self._override_on_timeout and
                  learner_output.inference_ms > 30.0):  # hard timeout
                safety_override = True
                safety_reason = "learner_inference_timeout"

            # Check velocity limits
            if not safety_override and learner_output is not None and learner_output.valid:
                vel = learner_output.velocity_flu
                speed = float(np.linalg.norm(vel))
                if speed > 3.0 or abs(learner_output.yaw_rate) > 3.0:
                    safety_override = True
                    safety_reason = "learner_command_out_of_bounds"
                if not np.all(np.isfinite(vel)):
                    safety_override = True
                    safety_reason = "learner_output_nan"

        # Determine final executed command
        if safety_override:
            self.safety_override_count += 1
            if expert_action_valid:
                return ("expert", True, safety_reason, None, None)  # use expert
            else:
                # Hover
                return ("safety", True, safety_reason,
                        np.zeros(3, dtype=np.float64), 0.0)

        if selected == "expert":
            self.expert_exec_count += 1
            return ("expert", False, "", None, None)  # use expert
        else:
            self.learner_exec_count += 1
            if learner_output is not None and learner_output.valid:
                return ("learner", False, "",
                        learner_output.velocity_flu.copy(),
                        learner_output.yaw_rate)
            else:
                # Learner invalid but no safety override configured
                # Fall back to expert
                self.safety_override_count += 1
                return ("expert", True, "learner_invalid_no_override", None, None)

    def get_output_dir(self):
        """Return DAgger output directory for current round."""
        rd = os.path.join(self._output_root, "round_{:03d}".format(self._round_id))
        os.makedirs(rd, exist_ok=True)
        return rd

    def stats_summary(self):
        return {
            "round_id": self._round_id,
            "beta": self._current_beta,
            "expert_exec_count": self.expert_exec_count,
            "learner_exec_count": self.learner_exec_count,
            "safety_override_count": self.safety_override_count,
            "invalid_label_count": self.invalid_label_count,
        }
