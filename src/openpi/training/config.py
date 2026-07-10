import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias, List, Tuple

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.shared.nnx_utils as nnx_utils
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.libero_rlds_dataset as libero_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # If true, the torch data loader will use a WeightedRandomSampler that upweights
    # frames with large action magnitude (and so naturally upweights grasp/place frames
    # over slow approach frames). Only takes effect for non-RLDS LeRobot datasets and
    # when not running PyTorch DDP. See `_build_phase_weights` in data_loader.py.
    phase_weighted_sampling: bool = False
    # Baseline weight assigned to every frame regardless of action magnitude.
    phase_weight_baseline: float = 0.2
    # Multiplier on the normalized action magnitude component of the weight.
    phase_weight_alpha: float = 1.0

    # Per-source mix sampling for co-training-mixture experiments. When
    # mix_fraction is set, frames whose LeRobot task string contains any of
    # mix_task_keywords form the MIX source and are sampled with expected
    # fraction mix_fraction; all other frames get the remaining 1-mix_fraction.
    # Decouples the mix ratio from episode counts in the combined repo (30 mix
    # demos can be sampled at 50%) and composes with phase_weighted_sampling.
    # Same constraints as phase weighting (non-RLDS LeRobot, no DDP). See
    # `_build_mix_weights` in data_loader.py.
    mix_fraction: float | None = None
    mix_task_keywords: Sequence[str] = ()

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    datasets: List[Tuple[str, float]] | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    dataset_class: type | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    env_action_dim: int = 7
    phase_weighted_sampling: bool = False
    phase_weight_baseline: float = 0.2
    phase_weight_alpha: float = 1.0
    # Per-source mix sampling (see DataConfig.mix_fraction / mix_task_keywords).
    mix_fraction: float | None = None
    mix_task_keywords: Sequence[str] = ()
    # Train-time augmentation. Applied inside the repack group, so it runs during
    # dataset iteration (training + compute_norm_stats) but NOT during inference.
    # Goal: robustify BC against closed-loop covariate shift by perturbing the
    # observations the policy is trained to predict from. Defaults disable.
    state_noise_std: float = 0.0
    image_brightness_jitter: float = 0.0
    image_contrast_jitter: float = 0.0
    image_saturation_jitter: float = 0.0

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_inputs: list[_transforms.DataTransformFn] = [
            _transforms.RepackTransform(
                {
                    "observation/image": "image",
                    "observation/wrist_image": "wrist_image",
                    "observation/state": "state",
                    "actions": "actions",
                    "prompt": "prompt",
                }
            )
        ]
        if self.state_noise_std > 0.0:
            repack_inputs.append(_transforms.StateGaussianNoise(sigma=self.state_noise_std))
        if max(self.image_brightness_jitter, self.image_contrast_jitter, self.image_saturation_jitter) > 0.0:
            repack_inputs.append(
                _transforms.ImageColorJitter(
                    brightness=self.image_brightness_jitter,
                    contrast=self.image_contrast_jitter,
                    saturation=self.image_saturation_jitter,
                )
            )
        repack_transform = _transforms.Group(inputs=repack_inputs)

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs(action_dim=self.env_action_dim)],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            phase_weighted_sampling=self.phase_weighted_sampling,
            phase_weight_baseline=self.phase_weight_baseline,
            phase_weight_alpha=self.phase_weight_alpha,
            mix_fraction=self.mix_fraction,
            mix_task_keywords=tuple(self.mix_task_keywords),
        )

@dataclasses.dataclass(frozen=True)
class RLDSLiberoDataConfig(DataConfigFactory):
    rlds_data_dir: str | None = None
    dataset_class = libero_rlds_dataset.LiberoRldsDataset
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/wrist_image": "observation/wrist_image",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            dataset_class=self.dataset_class,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    datasets: List[Tuple[str, float]] | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            datasets=self.datasets,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "/gscratch/scrubbed/arhanj/openpi/assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "/gpfs/scrubbed/arhanj/openpi/checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # If true, will save the train state to the checkpoint directory.
    save_train_state: bool = True
    # specific checkpoints to keep
    specific_checkpoints_to_keep: List[int] | None = dataclasses.field(default_factory=lambda: [100, 200, 500])

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

    TrainConfig(
        name="pi05_droid_jointpos_cotrain",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True, action_dim=8),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

    TrainConfig(
        name="pi05_droid_jointpos_nocotrain",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

     
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # LoRA fine-tune of pi05_libero on the local CompoSuite box+no-obstacle pilot
    # dataset (sanity-run; see docs/phase_2e_2f_report.md). Single 24GB GPU friendly.
    TrainConfig(
        name="pi05_libero_composuite_box_pilot",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_box_objective_v1",
            assets=AssetsConfig(
                assets_dir="/home/amy/.cache/openpi/openpi-assets/checkpoints/pi05_libero/assets",
                asset_id="physical-intelligence/libero",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=2_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/amy/.cache/openpi/openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=2_000,
        save_interval=500,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=True,
        wandb_enabled=False,
        checkpoint_base_dir="/home/amy/Projects/openpi/checkpoints",
        assets_base_dir="/home/amy/Projects/openpi/assets",
    ),
    # LoRA fine-tune for CompoSuite transfer tasks in the LIBERO pipeline.
    # For the initial plate-transfer sanity run, copy the LeRobot dataset to
    # $HF_LEROBOT_HOME/composuite_plate_transfer_sanity before running
    # compute_norm_stats.py / train.py on a cluster machine.
    # Norm stats: this config loads stats freshly computed on the local CompoSuite
    # dataset from <assets_base_dir>/<name>/<asset_id>/norm_stats.json — i.e.
    # assets/pi05_libero_composuite_transfer/composuite_plate_transfer_sanity/.
    # Do NOT reuse the bundled LIBERO stats: action distribution is heavily
    # asymmetric in scripted CompoSuite demos, and eef_z / wrist axis-angle live
    # near or past the LIBERO q01/q99 envelope (side-grasp is OOD for LIBERO).
    TrainConfig(
        name="pi05_libero_composuite_transfer",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_transfer_sanity",
            assets=AssetsConfig(
                asset_id="composuite_plate_transfer_sanity",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        # LR schedule, EMA, and step count mirror the pi05_libero reference
        # config. batch_size is set to 32 as a reasonable single-large-GPU
        # default (the reference uses 256 across multiple GPUs); bump it up
        # on a single 80 GB+ card or shard with fsdp_devices on multi-GPU.
        # Drop to 8 with ema_decay=None if running on 24 GB.
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        # Absolute results/ paths so compute_norm_stats.py (which builds the
        # config purely from its name, with no --assets-base-dir override) writes
        # norm stats to the SAME directory train.py reads from. Previously this
        # was relative ("assets"), so compute wrote to repos/openpi/assets while
        # train_pi05_libero.sh passed --assets-base-dir results/assets -- train
        # then silently skipped the missing stats (config._load_norm_stats), and
        # the CompoSuite-specific OOD norm stats never reached training.
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Forgetting / rehearsal experiment: identical recipe to v1
    # (pi05_libero_composuite_transfer) -- SAME model, LR schedule (warmup 10k ->
    # flat 5e-5), batch 32, EMA, 30k steps -- so the only variable vs the
    # plate-only baseline is the training mix. Dataset = the 100 plate demos
    # PLUS 30 box trash_can demos (rehearsal anchor), built by
    # experiments/plate_transfer/prepare_combined_plate_box.sh into the LeRobot
    # repo composuite_plate_box_trashcan. trash_can chosen over pick_and_place
    # as the rehearsal target: base pi0.5_libero sits at a MODERATE success rate
    # on it (~0.4-0.6), so there is a real, fragile skill to lose under plate-only
    # LoRA -- a more sensitive forgetting/preservation signal than near-ceiling
    # pick_and_place. Norm stats are computed over the UNION.
    TrainConfig(
        name="pi05_libero_composuite_plate_box_trashcan",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Fix experiment (1) -- norm-stats: SAME 100-plate + 30-trash_can dataset
    # (composuite_plate_box_trashcan) as pi05_libero_composuite_plate_box_trashcan,
    # but normalized with the PLATE-ONLY norm stats instead of the union. Tests whether
    # the union's ~30-38% wider translation-action std (box top-grasp variance), which
    # compresses plate's side-grasp deltas in normalized space, is what suppressed plate
    # learning (mixed model under-acts: leaves plate ~21cm from goal, 0/20 even in-dist).
    # The plate-only norm_stats.json is copied into this config's asset dir before train.
    # keep_period=5000 bounds checkpoint disk (diagnostic run; eval at milestones).
    TrainConfig(
        name="pi05_libero_composuite_plate_box_trashcan_platenorm",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=5000,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Fix experiment (2) -- data ratio: 100 plate + only 5 trash_can demos (vs 30), so the
    # box footprint is ~5% of the mix -- barely perturbs plate while (hopefully) still
    # anchoring the box skill (box generalized from little signal before). Dataset
    # composuite_plate_box5_trashcan (built by prepare_combined_plate_box.sh with a 5-demo
    # box subset); norm stats over this plate-dominated union. keep_period=5000.
    TrainConfig(
        name="pi05_libero_composuite_plate_box5_trashcan",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box5_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box5_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=5000,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Fix experiment (3) -- LoRA capacity: SAME 100-plate + 30-trash_can dataset and UNION
    # norm stats as pi05_libero_composuite_plate_box_trashcan (the failing rehearsal run),
    # but DOUBLES the LoRA rank (paligemma 16->32, action expert 32->64). Confirms whether
    # the mixed model's plate failure is adapter capacity. Expected: NO -- plate-only with
    # rank-16 also barely learns plate, so capacity isn't the binding constraint -- but we
    # run it to rule it out. Union norm_stats.json is copied into this config's asset dir.
    TrainConfig(
        name="pi05_libero_composuite_plate_box_trashcan_biglora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_r32",
            action_expert_variant="gemma_300m_lora_r64",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora_r32",
            action_expert_variant="gemma_300m_lora_r64",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=5000,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Paper-matched replay run (Zhu et al. 2026, real-world continual VLA). Combines the
    # best-practice levers: (a) buffer size B=0.2 -> 20 box trash_can demos per 100 plate;
    # (b) consistent/anchored norm stats (their "Strategy-I") -- the PLATE-ONLY norm stats
    # are copied into this config's asset dir (not the diluted union); (c) short warmup +
    # cosine decay 5e-5 -> 5e-6 (their recipe) instead of the 10k flat warmup. Goal: ONE
    # model that does plate (new, hard/OOD task -- expect ~plate-only ceiling, not mastery)
    # AND retains box (old skill, preserved by replay; trash_can replay also generalizes to
    # box pick&place). Differs from the paper: LoRA (not full SFT), batch 32 (not 128).
    TrainConfig(
        name="pi05_libero_composuite_plate_box20_papermatch",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box20_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box20_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=300,
            peak_lr=5e-5,
            decay_steps=6_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.998,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=6_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # ------------------------------------------------------------------
    # ICRA-2027 interference study: FROZEN PROTOCOL configs (Blocks A-G).
    # Recipe pre-registered in experiments/interference/PROTOCOL.md: LoRA r16,
    # batch 32, EMA 0.998, warmup 300 -> peak 5e-5 -> cosine to 5e-6 over 10k,
    # save every 2k, retimed single-prompt datasets, union quantile norm stats.
    # Block A: new-task-only baselines.
    *[
        TrainConfig(
            name=f"pi05_libero_protocol_{tag}",
            model=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                discrete_state_input=False,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ),
            data=LeRobotLiberoDataConfig(
                repo_id=repo,
                assets=AssetsConfig(asset_id=repo),
                base_config=DataConfig(prompt_from_task=True),
                extra_delta_transform=False,
            ),
            batch_size=32,
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=300,
                peak_lr=5e-5,
                decay_steps=10_000,
                decay_lr=5e-6,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            ema_decay=0.998,
            freeze_filter=pi0_config.Pi0Config(
                pi05=True,
                action_horizon=10,
                discrete_state_input=False,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
            ).get_freeze_filter(),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_libero/params"
            ),
            num_train_steps=10_000,
            save_interval=2_000,
            log_interval=10,
            keep_period=None,
            specific_checkpoints_to_keep=[],
            save_train_state=False,
            wandb_enabled=True,
            checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
            assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
        )
        for tag, repo in [
            ("plate", "composuite_plate_protocol"),
            ("dumbbell_tc", "composuite_dumbbell_tc_protocol"),
            ("box_shelf", "composuite_box_shelf_protocol"),
        ]
    ],
    # Phase B (sequential continuation, per Zhu et al. 2026). Init from the PLATE-COMPETENT
    # plate-only 10k checkpoint (= Phase A: the side-grasp wrist primitive already formed --
    # 90% side-grasp, 75% carry), then GENTLY recover box via replay while rehearsing plate.
    # Sequential staging avoids the joint-training domination that killed the wrist in the
    # papermatch CO-training run (which started from the box-competent base and never grew the
    # side-grasp). Low LR + short, to disturb the plate weights as little as possible; plate-
    # anchored norm stats (Strategy-I, copied into this config's asset dir). Watch the wrist/
    # carry stages vs box recovery at each checkpoint.
    TrainConfig(
        name="pi05_libero_composuite_plate_box20_phaseB",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_box20_trashcan",
            assets=AssetsConfig(
                asset_id="composuite_plate_box20_trashcan",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        # Gentle continuation: low peak LR, tiny warmup, decay -- recover box without
        # large weight moves that would erode the plate side-grasp.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=1e-5,
            decay_steps=4_000,
            decay_lr=1e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        # Init from the Phase-A (plate-only) 10k checkpoint, NOT the base.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/pajak/compositional-learning-vla/results/checkpoints/"
            "pi05_libero_composuite_transfer_v2/plate_doubleflip_blue_fixedlr/10000/params"
        ),
        num_train_steps=4_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # Stronger-recipe re-run of pi05_libero_composuite_transfer. The v1 sanity run
    # underfit (25% in-distribution on its own training layouts). Diagnosis: v1
    # copied the reference's batch-256 schedule -- warmup 10_000 (a THIRD of a 30k
    # run spent ramping up) at a flat low peak_lr=5e-5. This v2 keeps everything
    # else identical (same data, batch 32, 30k steps, EMA) and changes ONLY the LR
    # schedule: short warmup, 2x peak LR, and an actual cosine decay across the run.
    TrainConfig(
        name="pi05_libero_composuite_transfer_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_transfer_sanity",
            assets=AssetsConfig(
                asset_id="composuite_plate_transfer_sanity",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        # Fixed LR matching the pi05_libero LIBERO fine-tune: warm up, then hold
        # FLAT at 5e-5 (peak == decay over a 1M-step horizon, so it never really
        # decays within the run). The previous 1e-4 -> 1e-5 cosine decay over 30k
        # trained slowly with no real improvement; this returns to the LIBERO recipe.
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=None,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    # v3 -- data-format experiment: identical recipe to transfer_v2 (the ~20%
    # plate baseline: same model, LoRA, batch 32, 10k-warmup flat 5e-5, EMA,
    # 30k steps, base pi05_libero init). The ONLY change is the training data:
    # the same 100 seed-11 trajectories (regenerated as plate_libero_framing_v2
    # _phases with per-frame phase labels), converted with --retime (merges the
    # P-controller convergence tails; median commanded |dpos| 0.12 -> 0.52,
    # ~3.7x fewer frames -- attacks the measured under-acting: rollouts stalled
    # ~21cm short) and --subtask-split mixed (each episode also emitted as 3
    # phase-labeled sub-task segments with their own prompts, matching pi0.5's
    # subtask-decomposition pretraining; the full-prompt episode is kept so
    # full-prompt eval stays in-distribution). Any eval delta vs transfer_v2 is
    # attributable purely to the data formatting.
    # Build the dataset with:
    #   EPISODES_DIR=.../plate_libero_framing_v2_phases/episodes \
    #   REPO_ID=composuite_plate_retimed_subtask \
    #   RETIME_ARGS="--subtask-split mixed" \
    #   bash experiments/plate_transfer/convert_demos_retimed.sh
    TrainConfig(
        name="pi05_libero_composuite_transfer_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_retimed_subtask",
            assets=AssetsConfig(
                asset_id="composuite_plate_retimed_subtask",
            ),
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=32,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=30_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=5000,
        specific_checkpoints_to_keep=[],
        save_train_state=False,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/results/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/results/assets",
    ),
    TrainConfig(
        name="pi05_libero_composuite_plate_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_transfer_n100",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=5_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=5_000,
        save_interval=500,
        log_interval=10,
        keep_period=1_000,
        specific_checkpoints_to_keep=[500, 1_000, 2_500, 5_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v2 plate-transfer LoRA. Restarts from the clean pi05_libero base (not the
    # bad 1500-step checkpoint), keeps the OSC_POSE 7-dim action space the prior
    # was trained on, uses the trimmed 300-demo dataset, runs phase-weighted
    # sampling, and trains for the full 10k-step cosine schedule.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n300_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=10_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=2_000,
        specific_checkpoints_to_keep=[1_000, 2_500, 5_000, 7_500, 10_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v3 plate-transfer LoRA. Same recipe as v2 but trained on the n500 dataset
    # (300 baseline OSC demos + ~183 demos with 0.08 rad uniform per-joint
    # initial-pose jitter). The jittered demos add off-trajectory coverage near
    # episode start to address the closed-loop "stuck in start-of-demo action"
    # failure mode observed in the v2 eval.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v3",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n500_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=10_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=2_000,
        specific_checkpoints_to_keep=[1_000, 2_500, 5_000, 7_500, 10_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v4 plate-transfer LoRA. Same data as v3 (n500: 300 baseline + ~183 jitter-init
    # demos) but adds training-time augmentation — Gaussian state noise + image
    # color/contrast/saturation jitter — to robustify against the closed-loop
    # covariate shift that left v3 still 0/5 at step 5k. State_noise_std=0.01 is
    # ~1cm EEF pos / ~0.6deg rotation. Image jitter 0.15 is conservative.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v4",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n500_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
            state_noise_std=0.01,
            image_brightness_jitter=0.15,
            image_contrast_jitter=0.15,
            image_saturation_jitter=0.15,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=2.5e-5,
            decay_steps=10_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=10_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=2_000,
        specific_checkpoints_to_keep=[1_000, 2_500, 5_000, 7_500, 10_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v5 plate-transfer LoRA. Same data + augmentation as v4 but scales the
    # training recipe to match the OpenPI "decent finetune" reference setup
    # described in the medium.com/@yananchen1116 VLA generalisation article
    # (batch=64, peak_lr=5e-5, warmup~10% of total, longer total schedule).
    # Hypothesis: v3/v4 may have been undertrained at batch=16 / 10k steps;
    # combining augmentation with proper-scale training should give the model
    # a real chance to learn closed-loop manipulation.
    # Effective batch=64 is achieved via data-parallel across 4 GPUs, so each
    # device sees 16 samples (same per-device load as v4).
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v5",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n500_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
            state_noise_std=0.01,
            image_brightness_jitter=0.15,
            image_contrast_jitter=0.15,
            image_saturation_jitter=0.15,
        ),
        batch_size=64,
        fsdp_devices=1,  # pure data-parallel; LoRA model fits on one L40
        # LR shape mirrors the reference VLA finetune (10k warmup / 1M decay /
        # 50k trained): 20% warmup of total trained, then near-constant peak LR
        # through the rest of training (cosine decay over 1M steps is nearly
        # flat over our 20k window).
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=4_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=20_000,
        save_interval=1_000,
        log_interval=10,
        keep_period=5_000,
        specific_checkpoints_to_keep=[2_500, 5_000, 10_000, 15_000, 20_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v6 quicktest. Clean A/B for the camera-position patch. Trains a minimal
    # LoRA on a 50-demo set collected with the LIBERO-standard agentview pose,
    # with NO augmentation and NO jittered-init demos so the only delta vs the
    # original v2 baseline is the camera framing. If success rate at step 3k
    # is materially > 0 (v2 was 0/5), camera framing was the dominant factor
    # and a full v6 retrain at scale is justified.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v6_quicktest",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n50_camfix_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
        ),
        batch_size=16,
        fsdp_devices=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=3_000,
        save_interval=500,
        log_interval=10,
        keep_period=1_000,
        specific_checkpoints_to_keep=[1_000, 2_000, 3_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v6_overfit. Train pi05_libero LoRA on the camera-fixed n50 dataset with
    # MAXIMUM memorization pressure (no augmentation, no phase weighting,
    # 20k steps at flat peak LR). If open-loop MAE on the same training demos
    # converges to ~zero, the train side of the pipeline is fundamentally
    # capable of learning this skill from images. Closed-loop success rate is
    # secondary: if even the overfit model produces non-zero success on
    # rollouts, we know the bottleneck is generalization, not the pipeline.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v6_overfit",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n50_camfix_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=False,
        ),
        batch_size=16,
        fsdp_devices=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=20_000,
        save_interval=2_000,
        log_interval=10,
        keep_period=4_000,
        specific_checkpoints_to_keep=[4_000, 10_000, 20_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    # v6_general. Same n50 data as v6_overfit but with regularisation enabled
    # (state-noise + image-jitter augmentation, phase-weighted sampling). Tests
    # whether the augmentation recipe meaningfully changes closed-loop success
    # on new env seeds vs the pure overfit run, on this small dataset.
    TrainConfig(
        name="pi05_libero_composuite_plate_lora_v6_general",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_osc_n50_camfix_trimmed",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=7,
            phase_weighted_sampling=True,
            phase_weight_baseline=0.2,
            phase_weight_alpha=1.0,
            state_noise_std=0.01,
            image_brightness_jitter=0.15,
            image_contrast_jitter=0.15,
            image_saturation_jitter=0.15,
        ),
        batch_size=16,
        fsdp_devices=1,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=20_000,
        save_interval=2_000,
        log_interval=10,
        keep_period=4_000,
        specific_checkpoints_to_keep=[4_000, 10_000, 20_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    TrainConfig(
        name="pi05_libero_composuite_plate_joint_lora_from1500",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_joint_replay_n100_trimmed_v2",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=8,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=5_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/pajak/compositional-learning-vla/checkpoints/"
            "pi05_libero_composuite_plate_joint_lora/plate_joint_trimmed_v2_lora/1500/params"
        ),
        num_train_steps=5_000,
        save_interval=500,
        log_interval=10,
        keep_period=1_000,
        specific_checkpoints_to_keep=[500, 1_000, 2_500, 5_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    TrainConfig(
        name="pi05_libero_composuite_plate_joint_lora",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="composuite_plate_joint_replay_n100_trimmed_v2",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
            env_action_dim=8,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=100,
            peak_lr=2.5e-5,
            decay_steps=5_000,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=10,
            discrete_state_input=False,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ).get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_libero/params"
        ),
        num_train_steps=5_000,
        save_interval=500,
        log_interval=10,
        keep_period=1_000,
        specific_checkpoints_to_keep=[500, 1_000, 2_500, 5_000],
        save_train_state=True,
        wandb_enabled=True,
        checkpoint_base_dir="/home/pajak/compositional-learning-vla/checkpoints",
        assets_base_dir="/home/pajak/compositional-learning-vla/assets",
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        name="pi0_fast_droid_jointpos",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

    ### START ENCODER FINETUNE CONFIGS ###
    TrainConfig(
        name="pi05_droid_jointpos_encoderfinetune",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_sim_cotrain_set_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/mnt/bigguy",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi05_droid_jointpos/params"),
        freeze_filter=nnx_utils.PathRegex(".*llm.*"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=5_000,
        batch_size=128,
        log_interval=100,
        save_interval=500,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_fast_droid_jointpos_encoderfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=10,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            rlds_data_dir="/mnt/bigguy",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_fast_droid_jointpos/params"),
        freeze_filter=nnx_utils.PathRegex(".*llm.*"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=240_000,
        batch_size=128,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    

    TrainConfig(
        name="pi0_droid_jointpos_encoderfinetune",
        model=pi0_config.Pi0Config(
            # action_dim=8, # leave as 32 default...
            action_horizon=15,
            max_token_len=100,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi0_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_sim_cotrain_set_dataset", 0.05),
                ("droid_sim_in_dist_dataset", 0.05),
                ("droid", 1.0),
            ],
            rlds_data_dir="/mnt/bigguy",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=5_000,
        batch_size=128,
        log_interval=100,
        save_interval=500,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_droid_jointpos",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[
                    _transforms.AbsoluteActions(_transforms.make_bool_mask(7, -1)),
                    droid_policy.DroidOutputs(),
                ],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="paligemma_binning_droid_jointpos_encoderfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8, 
            action_horizon=15, 
            max_token_len=600,
            fast_model_tokenizer=_tokenizer.BinningTokenizer,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/paligemma_binning_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_sim_cotrain_set_dataset", 0.05),
                ("droid_sim_in_dist_dataset", 0.05),
                ("droid", 0.9),
            ],
            rlds_data_dir="/mnt/bigguy",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/paligemma_binning_droid_jointpos/params"),
        freeze_filter=nnx_utils.PathRegex(".*llm.*"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=5_000,
        batch_size=64,
        log_interval=100,
        save_interval=500,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    ### END ENCODER FINETUNE CONFIGS ###

    TrainConfig(
        name="pi05_droid_jointpos_fullfinetune_test",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset:1.0.1", 1.0),
                # ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi05_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=20,
        keep_period=40,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),


    ### START FULL FINETUNE CONFIGS ###
    TrainConfig(
        name="pi05_droid_jointpos_fullfinetune",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi05_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="paligemma_binning_droid_jointpos_fullfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8, 
            action_horizon=15, 
            max_token_len=600,
            fast_model_tokenizer=_tokenizer.BinningTokenizer,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/paligemma_binning_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/paligemma_binning_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=64,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_droid_jointpos_fullfinetune",
        model=pi0_config.Pi0Config(
            # action_dim=8, # leave as 32 default...
            action_horizon=10,
            max_token_len=100,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi0_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_droid_jointpos_100k_fullfinetune",
        model=pi0_config.Pi0Config(
            # action_dim=8, # leave as 32 default...
            action_horizon=10,
            max_token_len=100,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi0_droid_jointpos_100k/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_droid_jointpos_100k/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    
    TrainConfig(
        name="pi0_fast_droid_jointpos_fullfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=10,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="BLANK",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets-simeval/pi0_fast_droid_jointpos/assets",
                asset_id="droid",
            ),
            datasets=[
                ("droid_near_domain_dataset", 0.1),
                ("droid", 0.9),
            ],
            rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_fast_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    ### END FULL FINETUNE CONFIGS ###

    ### START LIBERO FULLFINETUNE CONFIGS ###
    TrainConfig(
        name="pi05_droid_libero_fullfinetune",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=RLDSLiberoDataConfig(
            repo_id="libero_90",
            assets=AssetsConfig(
                # TODO: recompute assets
                # assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                # asset_id="droid",
            ),
            # rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            rlds_data_dir="/gscratch/scrubbed/arhanj/datasets",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi05_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=128,
        log_interval=100,
        save_interval=20_000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="paligemma_binning_droid_libero_fullfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8, 
            action_horizon=15, 
            max_token_len=600,
            fast_model_tokenizer=_tokenizer.BinningTokenizer,
        ),
        data=RLDSLiberoDataConfig(
            repo_id="libero_90",
            assets=AssetsConfig(
                # TODO: recompute assets
                # assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                # asset_id="droid",
            ),
            # rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            rlds_data_dir="/gscratch/scrubbed/arhanj/datasets",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/paligemma_binning_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=64,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_droid_libero_fullfinetune",
        model=pi0_config.Pi0Config(
            # action_dim=8, # leave as 32 default...
            action_horizon=10,
            max_token_len=100,
        ),
        data=RLDSLiberoDataConfig(
            repo_id="libero_90",
            assets=AssetsConfig(
                # TODO: recompute assets
                # assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                # asset_id="droid",
            ),
            # rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            rlds_data_dir="/gscratch/scrubbed/arhanj/datasets",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    TrainConfig(
        name="pi0_droid_libero_100k_fullfinetune",
        model=pi0_config.Pi0Config(
            # action_dim=8, # leave as 32 default...
            action_horizon=10,
            max_token_len=100,
        ),
        data=RLDSLiberoDataConfig(
            repo_id="libero_90",
            assets=AssetsConfig(
                # TODO: recompute assets
                # assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                # asset_id="droid",
            ),
            # rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            rlds_data_dir="/gscratch/scrubbed/arhanj/datasets",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_droid_jointpos_100k/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),

    
    TrainConfig(
        name="pi0_fast_droid_libero_fullfinetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=10,
            max_token_len=180,
        ),
        data=RLDSLiberoDataConfig(
            repo_id="libero_90",
            assets=AssetsConfig(
                # TODO: recompute assets
                # assets_dir="gs://openpi-assets-simeval/pi05_droid_jointpos/assets",
                # asset_id="droid",
            ),
            # rlds_data_dir="/gpfs/scrubbed/arhanj/datasets",
            rlds_data_dir="/gscratch/scrubbed/arhanj/datasets",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-simeval/pi0_fast_droid_jointpos/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=10_000,
        batch_size=128,
        log_interval=100,
        save_interval=1000,
        keep_period=1000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),


    ### END LIBERO FULLFINETUNE CONFIGS ###

    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    names = {}
    for config in _CONFIGS:
        if config.name in names:
            print(f"Config {config.name} already exists.")
        names[config.name] = config

    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
