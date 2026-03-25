# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

from ..fb_cpr_aux.model import FBcprAuxModel, FBcprAuxModelArchiConfig, FBcprAuxModelConfig
from ..nn_models import (
    BackwardArchiConfig,
    OneStepForwardArchiConfig,
    RewardNormalizerConfig,
)


class OneStepFBcprAuxModelArchiConfig(FBcprAuxModelArchiConfig):
    # Override f and b to default to the one-step variants
    f: OneStepForwardArchiConfig = OneStepForwardArchiConfig()
    b: BackwardArchiConfig = BackwardArchiConfig()


class OneStepFBcprAuxModelConfig(FBcprAuxModelConfig):
    name: tp.Literal["OneStepFBcprAuxModel"] = "OneStepFBcprAuxModel"
    archi: OneStepFBcprAuxModelArchiConfig = OneStepFBcprAuxModelArchiConfig()
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig()

    @property
    def object_class(self):
        return OneStepFBcprAuxModel


class OneStepFBcprAuxModel(FBcprAuxModel):
    config_class = OneStepFBcprAuxModelConfig
