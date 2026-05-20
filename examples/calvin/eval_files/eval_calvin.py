"""
Calvin Multi-Step Evaluation Script

Based on RoboFlamingo's evaluation protocol:
https://github.com/RoboFlamingo/RoboFlamingo/blob/main/robot_flamingo/eval/eval_utils.py

Evaluates a policy server on Calvin's long-horizon multi-task benchmark.
Measures success rate on chains of 1-5 consecutive tasks.

Usage:
    python examples/calvin/eval_calvin.py \
        --args.host 0.0.0.0 \
        --args.port 8000 \
        --args.dataset_path /path/to/calvin/task_D_D \
        --args.num_sequences 1000
"""

import copy
import contextlib
import dataclasses
import json
import logging
import os
import time
from collections import Counter, defaultdict
from math import pi
from pathlib import Path

import hydra
import numpy as np
try:
    import tyro
except ModuleNotFoundError:
    tyro = None

from omegaconf import OmegaConf
try:
    from termcolor import colored
except ModuleNotFoundError:
    def colored(text, *_args, **_kwargs):
        return text

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    class tqdm:
        def __init__(self, iterable, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_description(self, *_args, **_kwargs):
            return None

from deployment.model_server.tools import image_tools
from examples.LIBERO.eval_files.model2libero_interface import ModelClient

# from calvin_env.envs.play_table_env import get_env

_render_backend_env = os.environ.get("STARVLA_CALVIN_RENDER_BACKEND", "auto").strip().lower()
if _render_backend_env in {"egl", "gpu", "cuda", "fast", "auto"}:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "0"
else:
    os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EP_LEN = int(os.environ.get("CALVIN_EP_LEN", "360"))  # Max steps per task


def collect_plan(model, plans, subtask):
    try:
        plans[subtask].append((model.plan.cpu(), model.latent_goal.cpu()))
    except AttributeError:
        return


def count_success(results):
    if not results:
        return [0.0] * 5
    count = Counter(results)
    step_success = []
    for i in range(1, 6):
        n_success = sum(count[j] for j in reversed(range(i, 6)))
        step_success.append(n_success / len(results))
    return step_success


def print_and_save(results, sequences, log_dir, epoch=None):
    current_data = {}
    print(f"Results for Epoch {epoch}:")
    avg_seq_len = float(np.mean(results)) if results else 0.0
    chain_sr = {i + 1: float(sr) for i, sr in enumerate(count_success(results))}
    print(f"Average successful sequence length: {avg_seq_len}")
    print("Success rates for i instructions in a row:")
    for i, sr in chain_sr.items():
        print(f"{i}: {sr * 100:.1f}%")

    cnt_success = Counter()
    cnt_fail = Counter()

    for result, (_, sequence) in zip(results, sequences):
        for successful_tasks in sequence[:result]:
            cnt_success[successful_tasks] += 1
        if result < len(sequence):
            failed_task = sequence[result]
            cnt_fail[failed_task] += 1

    total = cnt_success + cnt_fail
    task_info = {}
    for task in total:
        task_info[task] = {"success": int(cnt_success[task]), "total": int(total[task])}
        print(f"{task}: {cnt_success[task]} / {total[task]} |  SR: {cnt_success[task] / total[task] * 100:.1f}%")

    data = {"avg_seq_len": avg_seq_len, "chain_sr": chain_sr, "task_info": task_info}
    current_data[str(epoch)] = data

    print()
    previous_data = {}
    try:
        with open(log_dir / "results.json", "r") as file:
            previous_data = json.load(file)
    except FileNotFoundError:
        pass
    json_data = {**previous_data, **current_data}
    with open(log_dir / "results.json", "w") as file:
        json.dump(json_data, file)
    if json_data:
        print(
            f"Best model: epoch {max(json_data, key=lambda x: json_data[x]['avg_seq_len'])} "
            f"with average sequences length of {max(map(lambda x: x['avg_seq_len'], json_data.values()))}"
        )


def get_log_dir(log_dir):
    if log_dir is not None:
        log_dir = Path(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = Path(__file__).parents[3] / "evaluation"
        if not log_dir.exists():
            log_dir = Path("/tmp/evaluation")
            os.makedirs(log_dir, exist_ok=True)
    print(f"logging to {log_dir}")
    return log_dir


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def fnv1a_32(text):
    value = 2166136261
    for byte in str(text).encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def get_env_state_for_initial_condition(initial_condition):
    robot_obs = np.array(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ]
    )
    block_rot_z_range = (pi / 2 - pi / 8, pi / 2 + pi / 8)
    block_slider_left = np.array([-2.40851662e-01, 9.24044687e-02, 4.60990009e-01])
    block_slider_right = np.array([7.03416330e-02, 9.24044687e-02, 4.60990009e-01])
    block_table = [
        np.array([5.00000896e-02, -1.20000177e-01, 4.59990009e-01]),
        np.array([2.29995412e-01, -1.19995140e-01, 4.59990010e-01]),
    ]

    seed = fnv1a_32(str(initial_condition.values()))
    with temp_seed(seed):
        np.random.shuffle(block_table)

        scene_obs = np.zeros(24)
        if initial_condition["slider"] == "left":
            scene_obs[0] = 0.28
        if initial_condition["drawer"] == "open":
            scene_obs[1] = 0.22
        if initial_condition["lightbulb"] == 1:
            scene_obs[3] = 0.088
        scene_obs[4] = initial_condition["lightbulb"]
        scene_obs[5] = initial_condition["led"]

        if initial_condition["red_block"] == "slider_right":
            scene_obs[6:9] = block_slider_right
        elif initial_condition["red_block"] == "slider_left":
            scene_obs[6:9] = block_slider_left
        else:
            scene_obs[6:9] = block_table[0]
        scene_obs[11] = np.random.uniform(*block_rot_z_range)

        if initial_condition["blue_block"] == "slider_right":
            scene_obs[12:15] = block_slider_right
        elif initial_condition["blue_block"] == "slider_left":
            scene_obs[12:15] = block_slider_left
        elif initial_condition["red_block"] == "table":
            scene_obs[12:15] = block_table[1]
        else:
            scene_obs[12:15] = block_table[0]
        scene_obs[17] = np.random.uniform(*block_rot_z_range)

        if initial_condition["pink_block"] == "slider_right":
            scene_obs[18:21] = block_slider_right
        elif initial_condition["pink_block"] == "slider_left":
            scene_obs[18:21] = block_slider_left
        else:
            scene_obs[18:21] = block_table[1]
        scene_obs[23] = np.random.uniform(*block_rot_z_range)

    return robot_obs, scene_obs


def make_image_sequence_clip(frames, fps):
    from moviepy.editor import ImageSequenceClip

    return ImageSequenceClip(frames, fps=fps)


def make_close_idempotent(env):
    original_close = env.close
    closed = False

    def close_once():
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            original_close()
        except Exception as exc:
            logger.debug("Ignoring Calvin env close error after shutdown: %r", exc)

    env.close = close_once
    return env


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    use_ddim: bool = True
    num_ddim_steps: int = 10
    pretrained_path: str = ""
    unnorm_key: str = ""

    #################################################################################################################
    # Calvin environment-specific parameters
    #################################################################################################################
    dataset_path: str = "/path/to/calvin/task_D_D"  # Path to Calvin dataset
    calvin_config_path: str = "/path/to/calvin/calvin_models/conf"
    eval_sequences_path: str = "/path/to/calvin/eval_sequences.json"
    num_sequences: int = 1000  # Number of evaluation sequences
    max_steps_per_task: int = EP_LEN  # Official CALVIN uses 360; lower this only for quick smoke checks.
    sequence_start: int = 0  # Shard start index for parallel CALVIN evaluation.
    sequence_end: int = -1  # Optional exclusive end index; use with sequence_start for contiguous worker shards.
    sequence_stride: int = 1  # Shard stride; use one worker per stride slot.
    num_workers: int = 1  # For future multi-process support
    seed: int = 0
    create_plan_tsne: bool = False

    #################################################################################################################
    # Evaluation settings
    #################################################################################################################
    debug: bool = False  # Save debug videos
    eval_log_dir: str = "tmp/calvin/eval_logs"  # Path to save evaluation logs and videos
    reset: bool = False  # If True, reset robot state between tasks (easier)
    diverse_inst: bool = False  # Use diverse instructions (zero-shot generalization)


class CalvinPolicyClient:
    """Wrapper around websocket client with Calvin-specific preprocessing."""

    def __init__(
        self,
        host: str,
        port: int,
        resize_size: int = 224,
        replan_steps: int = 5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        pretrained_path: str = "",
        unnorm_key: str = "",
    ):
        self.client = ModelClient(
            unnorm_key=(unnorm_key or None),
            policy_setup="franka",
            horizon=0,
            action_ensemble=False,
            use_ddim=use_ddim,
            num_ddim_steps=num_ddim_steps,
            host=host,
            port=port,
        )
        self.resize_size = resize_size
        self.replan_steps = replan_steps
        self.pretrained_path = pretrained_path
        self.step_count = 0
        self.server_metadata = getattr(self.client, "_server_metadata", {}) or {}
        self.state_dim = self._resolve_state_dim(self.server_metadata)
        self.state_slice = os.environ.get("STARVLA_CALVIN_STATE_SLICE", "first").strip().lower()
        self._state_adapt_warned = False
        print(
            "[starvla-calvin-eval] policy metadata "
            f"action_dim={self.server_metadata.get('action_dim')} "
            f"state_dim={self.server_metadata.get('state_dim')} "
            f"chunk={self.server_metadata.get('action_chunk_size')} "
            f"client_state_dim={self.state_dim} "
            f"state_slice={self.state_slice}",
            flush=True,
        )

    def reset(self):
        """Reset action plan buffer."""
        self.step_count = 0
        self.client.reset(None)

    @staticmethod
    def _resolve_state_dim(metadata: dict) -> int:
        env_value = os.environ.get("STARVLA_CALVIN_STATE_DIM", "auto").strip().lower()
        if env_value not in {"", "auto", "default", "metadata"}:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except ValueError:
                logger.warning("Ignoring invalid STARVLA_CALVIN_STATE_DIM=%r; using metadata/default", env_value)
        for key in ("state_dim", "proprio_dim"):
            try:
                parsed = int(metadata.get(key))
                if parsed > 0:
                    return parsed
            except (TypeError, ValueError):
                pass
        return 7

    def _state_from_obs(self, robot_obs) -> np.ndarray:
        raw = np.asarray(robot_obs, dtype=np.float32).reshape(-1)
        target = int(self.state_dim or 7)
        if raw.shape[0] == target:
            return raw
        if raw.shape[0] > target:
            if self.state_slice in {"first", "head", "eef", "eef7"}:
                state = raw[:target]
            elif self.state_slice in {"last", "tail"}:
                state = raw[-target:]
            else:
                logger.warning("Unknown STARVLA_CALVIN_STATE_SLICE=%r; using first %d dims", self.state_slice, target)
                state = raw[:target]
        else:
            state = np.pad(raw, (0, target - raw.shape[0]), mode="constant")
        if not self._state_adapt_warned:
            logger.warning(
                "CALVIN client adapted robot_obs state from %d to %d dims using slice=%s. "
                "This fixes checkpoint/client state_dim mismatches early.",
                raw.shape[0],
                target,
                self.state_slice,
            )
            self._state_adapt_warned = True
        return state.astype(np.float32, copy=False)

    def step(self, obs: dict, lang_annotation: str) -> np.ndarray:
        """
        Query policy for action given observation and language instruction.

        Args:
            obs: Calvin observation dict with keys:
                - rgb_obs: dict with 'rgb_static' (200x200x3) and 'rgb_gripper' (84x84x3)
                - robot_obs: (15,) proprioceptive state [ee_pos(3), ee_ori(3), gripper(2), joint_pos(7)]
            lang_annotation: Natural language task description
            get_action: If True, query model for new action chunk

        Returns:
            action: (7,) array [dx, dy, dz, droll, dpitch, dyaw, gripper]
        """
        # Preprocess images
        rgb_static = obs["rgb_obs"]["rgb_static"]  # (200, 200, 3) uint8
        rgb_gripper = obs["rgb_obs"]["rgb_gripper"]  # (84, 84, 3) uint8

        # Resize and pad images
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(rgb_static, self.resize_size, self.resize_size))
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(rgb_gripper, self.resize_size, self.resize_size)
        )

        # Prepare input for policy server (aligned with eval_libero)
        example = {
            "image": [image, wrist_image],
            "lang": lang_annotation,
            "state": self._state_from_obs(obs["robot_obs"])[None, :],
        }

        # Query model
        model_output = self.client.step(example=example, step=self.step_count)
        raw_action = model_output["raw_action"]
        world_vector = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
        rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
        open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)

        action = np.concatenate([world_vector, rotation_delta, open_gripper], axis=0).astype(np.float32)
        self.step_count += 1
        return action


def _normalize_render_backend(value: str) -> str:
    backend = (value or "auto").strip().lower()
    if backend in {"egl", "gpu", "cuda", "fast", "auto"}:
        return "egl"
    if backend in {"direct", "osmesa", "safe", "cpu"}:
        return "direct"
    logger.warning("Unknown STARVLA_CALVIN_RENDER_BACKEND=%r; using direct/osmesa", value)
    return "direct"


def _configure_egl_device(render_gpu: str | None) -> None:
    if not render_gpu:
        return
    if "EGL_VISIBLE_DEVICES" in os.environ or "EGL_VISIBLE_DEVICE" in os.environ:
        return
    try:
        cuda_id = int(str(render_gpu).split(",")[0])
    except ValueError:
        logger.warning("Ignoring invalid STARVLA_CALVIN_RENDER_GPU=%r", render_gpu)
        return
    try:
        from calvin_env.utils.utils import get_egl_device_id

        egl_id = get_egl_device_id(cuda_id)
    except Exception as exc:
        logger.warning("Could not map CUDA GPU %s to an EGL device: %r; trying the CUDA id directly", cuda_id, exc)
        egl_id = cuda_id
    os.environ["EGL_VISIBLE_DEVICES"] = str(egl_id)
    os.environ["EGL_VISIBLE_DEVICE"] = str(egl_id)
    logger.info("CALVIN EGL render device %s selected for CUDA GPU %s", egl_id, cuda_id)


def make_env(dataset_path: str):
    """Initialize Calvin environment without tactile sensor (to avoid OpenGL issues)."""
    dataset_root = Path(dataset_path)
    val_folder = dataset_root if (dataset_root / ".hydra" / "merged_config.yaml").exists() else dataset_root / "validation"

    # Load config and disable tactile sensor to avoid pyrender/OpenGL conflicts
    from omegaconf import OmegaConf

    config_path = val_folder / ".hydra" / "merged_config.yaml"
    cfg = OmegaConf.load(config_path)
    render_backend = _normalize_render_backend(os.environ.get("STARVLA_CALVIN_RENDER_BACKEND", "auto"))
    use_egl = render_backend == "egl"
    if use_egl:
        _configure_egl_device(os.environ.get("STARVLA_CALVIN_RENDER_GPU"))
    cfg.env.use_egl = bool(use_egl)
    logger.info(
        "Calvin render backend=%s use_egl=%s render_gpu=%s config=%s",
        render_backend,
        cfg.env.use_egl,
        os.environ.get("STARVLA_CALVIN_RENDER_GPU", "auto"),
        config_path,
    )

    # The StarVLA policy consumes only static and gripper RGB. Avoid tactile or extra cameras.
    if hasattr(cfg.env, "cameras"):
        wanted_cameras = {"static", "gripper"}
        new_cameras = OmegaConf.create({k: v for k, v in cfg.env.cameras.items() if k in wanted_cameras})
        if new_cameras:
            cfg.env.cameras = new_cameras

    # Initialize environment with modified config
    import hydra

    env = hydra.utils.instantiate(cfg.env, show_gui=False, use_vr=False, use_scene_info=True, use_egl=bool(use_egl))
    make_close_idempotent(env)

    return env


def load_lang_task(dataset_path: str) -> dict:
    """Load language annotations and task oracle for Calvin validation set."""
    conf_dir = Path(dataset_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    return val_annotations, task_oracle


def evaluate_policy_ddp(
    policy,
    env,
    epoch,
    calvin_conf_path,
    eval_sequences_path,
    num_sequences,
    max_steps_per_task=EP_LEN,
    sequence_start=0,
    sequence_end=-1,
    sequence_stride=1,
    eval_log_dir=None,
    debug=False,
    create_plan_tsne=False,
    reset=False,
    diverse_inst=False,
):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: Must implement methods of CalvinBaseModel.
        env: (Wrapped) calvin env.
        epoch:
        eval_log_dir: Path where to log evaluation results. If None, logs to /tmp/evaluation/
        debug: If True, show camera view and debug info.
        create_plan_tsne: Collect data for TSNE plots of latent plans (does not work for your custom model)

    Returns:
        Dictionary with results
    """
    conf_dir = Path(calvin_conf_path)
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)

    # val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")
    if diverse_inst:
        with open("/mnt/bn/robotics/lxh/robot-flamingo/lang_annotation_cache.json", "r") as f:
            val_annotations = json.load(f)
    else:
        val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)
    with open(eval_sequences_path, "r") as f:
        all_eval_sequences = json.load(f)

    sequence_start = max(int(sequence_start or 0), 0)
    sequence_end = -1 if sequence_end is None else int(sequence_end)
    sequence_stride = max(int(sequence_stride or 1), 1)
    if sequence_end >= 0:
        sequence_end = min(max(sequence_end, sequence_start), len(all_eval_sequences))
        selected_eval_sequences = all_eval_sequences[sequence_start:sequence_end]
        indexed_eval_sequences = list(enumerate(selected_eval_sequences, start=sequence_start))
        eval_items = indexed_eval_sequences[::sequence_stride]
        selected_desc = f"[{sequence_start}, {sequence_end})"
    else:
        if num_sequences is None or num_sequences <= 0:
            selected_eval_sequences = all_eval_sequences
        else:
            selected_eval_sequences = all_eval_sequences[: min(num_sequences, len(all_eval_sequences))]
        indexed_eval_sequences = list(enumerate(selected_eval_sequences))
        eval_items = indexed_eval_sequences[sequence_start::sequence_stride]
        selected_desc = f"first {len(selected_eval_sequences)}"
    logger.info(
        "Evaluating %d CALVIN sequences from %s selected / %d official, shard_start=%d, shard_stride=%d, max_steps_per_task=%d",
        len(eval_items),
        selected_desc,
        len(all_eval_sequences),
        sequence_start,
        sequence_stride,
        max_steps_per_task,
    )
    if not eval_items:
        logger.warning("No CALVIN sequences assigned to shard start=%d stride=%d", sequence_start, sequence_stride)
        with open(eval_log_dir / "sequence_progress.json", "w") as f:
            json.dump({"complete": True, "results": [], "num_sequences": 0, "total_sequences": 0, "sequences": []}, f, indent=2, sort_keys=True)
        with open(eval_log_dir / "sequence_results.json", "w") as f:
            json.dump({"results": [], "num_sequences": 0, "sequences": []}, f, indent=2, sort_keys=True)
        return []
    # device_num = int(torch.distributed.get_world_size())
    # device_id = torch.distributed.get_rank()
    # assert num_sequences % device_num == 0
    # interval_len = int(num_sequences // device_num)
    # eval_sequences = eval_sequences[device_id*interval_len:min((device_id+1)*interval_len, num_sequences)]
    results = []
    raw_results = []
    plans = defaultdict(list)
    total_sequences = len(eval_items)
    progress_start_time = time.time()

    def write_sequence_progress(complete: bool = False) -> None:
        with open(eval_log_dir / "sequence_progress.json", "w") as f:
            json.dump(
                {
                    "complete": bool(complete),
                    "results": [int(item) for item in results],
                    "num_sequences": len(results),
                    "total_sequences": total_sequences,
                    "sequences": raw_results,
                },
                f,
                indent=2,
                sort_keys=True,
            )

    if not debug:
        progress_iter = tqdm(eval_items, position=0, leave=True)
    else:
        progress_iter = eval_items

    for original_sequence_i, (initial_state, eval_sequence) in progress_iter:
        result = evaluate_sequence(
            env,
            policy,
            task_oracle,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
            eval_log_dir,
            original_sequence_i,
            reset=reset,
            diverse_inst=diverse_inst,
            max_steps_per_task=max_steps_per_task,
        )
        results.append(result)
        raw_results.append(
            {
                "sequence_index": int(original_sequence_i),
                "initial_state": initial_state,
                "sequence": eval_sequence,
                "success_count": int(result),
            }
        )
        write_sequence_progress(False)
        if not debug:
            chain_success = count_success(results)
            progress_iter.set_description(
                " ".join([f"{i + 1}/5 : {v * 100:.1f}% |" for i, v in enumerate(chain_success)]) + "|"
            )
            elapsed = time.time() - progress_start_time
            avg_seq_len = float(np.mean(results)) if results else 0.0
            chain_text = ",".join(f"{i + 1}:{v * 100:.1f}%" for i, v in enumerate(chain_success))
            print(
                "[progress] "
                f"sequence_index={original_sequence_i} "
                f"local={len(results)}/{total_sequences} "
                f"result={int(result)} "
                f"avg_seq_len={avg_seq_len:.3f} "
                f"chain_sr={chain_text} "
                f"elapsed_sec={elapsed:.1f}",
                flush=True,
            )

    def merge_multi_list(res):
        tmp = []
        for l in res:
            tmp.extend(l)
        return tmp

    def extract_iter_from_tqdm(tqdm_iter):
        return [_ for _ in tqdm_iter]

    # if create_plan_tsne:
    #     create_tsne(plans, eval_log_dir, epoch)

    print_and_save(results, [item for _, item in eval_items], eval_log_dir, epoch)
    write_sequence_progress(True)
    with open(eval_log_dir / "sequence_results.json", "w") as f:
        json.dump(
            {
                "results": [int(result) for result in results],
                "num_sequences": len(results),
                "sequences": raw_results,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    return results


def evaluate_sequence(
    env,
    policy,
    task_checker,
    initial_state,
    eval_sequence,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    sequence_i=-1,
    reset=False,
    diverse_inst=False,
    max_steps_per_task=EP_LEN,
):
    """
    Evaluates a sequence of language instructions.
    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter = 0
    if debug:
        time.sleep(1)
        print()
        print()
        print(f"Evaluating sequence: {' -> '.join(eval_sequence)}")
        print("Subtask: ", end="")
    for subtask_i, subtask in enumerate(eval_sequence):
        if reset:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                robot_obs=robot_obs,
                scene_obs=scene_obs,
                diverse_inst=diverse_inst,
                max_steps_per_task=max_steps_per_task,
            )
        else:
            success = rollout(
                env,
                policy,
                task_checker,
                subtask,
                val_annotations,
                plans,
                debug,
                eval_log_dir,
                subtask_i,
                sequence_i,
                diverse_inst=diverse_inst,
                max_steps_per_task=max_steps_per_task,
            )
        if success:
            success_counter += 1
        else:
            return success_counter
    return success_counter


def rollout(
    env,
    policy,
    task_oracle,
    subtask,
    val_annotations,
    plans,
    debug,
    eval_log_dir="",
    subtask_i=-1,
    sequence_i=-1,
    robot_obs=None,
    scene_obs=None,
    diverse_inst=False,
    max_steps_per_task=EP_LEN,
):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).
    """
    if debug:
        print(f"{subtask} ", end="")
        time.sleep(0.5)
    if robot_obs is not None and scene_obs is not None:
        env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
    obs = env.get_obs()
    # get lang annotation for subtask
    if diverse_inst:
        lang_annotation = val_annotations[sequence_i][subtask_i]
    else:
        lang_annotation = val_annotations[subtask][0]
    lang_annotation = lang_annotation.split("\n")[0]
    if "\u2019" in lang_annotation:
        lang_annotation = lang_annotation.replace("\u2019", "'")
    policy.reset()
    start_info = env.get_info()

    if debug:
        img_queue = []

    for step in range(max_steps_per_task):

        action = policy.step(obs, lang_annotation)

        # Ensure action is writable (Calvin env modifies it in-place)
        if not action.flags.writeable:
            action = np.array(action, copy=True)
        action[-1] = 1 if action[-1] > 0 else -1

        obs, _, _, current_info = env.step(action)
        if debug:
            img_copy = copy.deepcopy(obs["rgb_obs"]["rgb_static"])
            img_queue.append(img_copy)
        if step == 0:
            # for tsne plot, only if available
            collect_plan(policy, plans, subtask)

        # check if current step solves a task
        current_task_info = task_oracle.get_task_info_for_set(start_info, current_info, {subtask})
        if len(current_task_info) > 0:
            if debug:
                print(colored("success", "green"), end=" ")
                img_clip = make_image_sequence_clip(img_queue, fps=30)
                img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-succ.gif"), fps=30)
            return True
    if debug:
        print(colored("fail", "red"), end=" ")
        img_clip = make_image_sequence_clip(img_queue, fps=30)
        img_clip.write_gif(os.path.join(eval_log_dir, f"{sequence_i}-{subtask_i}-{subtask}-fail.gif"), fps=30)
    return False


def main(args: Args):
    # args = tyro.cli(Args)

    policy = CalvinPolicyClient(
        args.host,
        args.port,
        args.resize_size,
        args.replan_steps,
        args.use_ddim,
        args.num_ddim_steps,
        pretrained_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
    )
    env = make_env(args.dataset_path)

    evaluate_policy_ddp(
        policy,
        env,
        0,
        args.calvin_config_path,
        args.eval_sequences_path,
        args.num_sequences,
        args.max_steps_per_task,
        args.sequence_start,
        args.sequence_end,
        args.sequence_stride,
        args.eval_log_dir,
        args.debug,
        args.create_plan_tsne,
        args.reset,
        args.diverse_inst,
    )


def parse_args_without_tyro() -> Args:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--args.host", dest="host", default=Args.host)
    parser.add_argument("--args.port", dest="port", type=int, default=Args.port)
    parser.add_argument("--args.resize-size", "--args.resize_size", dest="resize_size", type=int, default=Args.resize_size)
    parser.add_argument("--args.replan-steps", "--args.replan_steps", dest="replan_steps", type=int, default=Args.replan_steps)
    parser.add_argument("--args.use-ddim", "--args.use_ddim", dest="use_ddim", action="store_true", default=Args.use_ddim)
    parser.add_argument("--args.no-use-ddim", "--args.no_use_ddim", dest="use_ddim", action="store_false")
    parser.add_argument(
        "--args.num-ddim-steps",
        "--args.num_ddim_steps",
        dest="num_ddim_steps",
        type=int,
        default=Args.num_ddim_steps,
    )
    parser.add_argument("--args.pretrained-path", "--args.pretrained_path", dest="pretrained_path", default=Args.pretrained_path)
    parser.add_argument("--args.unnorm-key", "--args.unnorm_key", dest="unnorm_key", default=Args.unnorm_key)
    parser.add_argument("--args.dataset-path", "--args.dataset_path", dest="dataset_path", default=Args.dataset_path)
    parser.add_argument(
        "--args.calvin-config-path",
        "--args.calvin_config_path",
        dest="calvin_config_path",
        default=Args.calvin_config_path,
    )
    parser.add_argument(
        "--args.eval-sequences-path",
        "--args.eval_sequences_path",
        dest="eval_sequences_path",
        default=Args.eval_sequences_path,
    )
    parser.add_argument("--args.num-sequences", "--args.num_sequences", dest="num_sequences", type=int, default=Args.num_sequences)
    parser.add_argument(
        "--args.max-steps-per-task",
        "--args.max_steps_per_task",
        dest="max_steps_per_task",
        type=int,
        default=Args.max_steps_per_task,
    )
    parser.add_argument("--args.sequence-start", "--args.sequence_start", dest="sequence_start", type=int, default=Args.sequence_start)
    parser.add_argument("--args.sequence-end", "--args.sequence_end", dest="sequence_end", type=int, default=Args.sequence_end)
    parser.add_argument("--args.sequence-stride", "--args.sequence_stride", dest="sequence_stride", type=int, default=Args.sequence_stride)
    parser.add_argument("--args.num-workers", "--args.num_workers", dest="num_workers", type=int, default=Args.num_workers)
    parser.add_argument("--args.seed", dest="seed", type=int, default=Args.seed)
    parser.add_argument("--args.create-plan-tsne", "--args.create_plan_tsne", dest="create_plan_tsne", action="store_true")
    parser.add_argument("--args.debug", dest="debug", action="store_true")
    parser.add_argument("--args.eval-log-dir", "--args.eval_log_dir", dest="eval_log_dir", default=Args.eval_log_dir)
    parser.add_argument("--args.reset", dest="reset", action="store_true")
    parser.add_argument("--args.diverse-inst", "--args.diverse_inst", dest="diverse_inst", action="store_true")
    return Args(**vars(parser.parse_args()))


if __name__ == "__main__":
    if tyro is not None:
        tyro.cli(main)
    else:
        main(parse_args_without_tyro())
