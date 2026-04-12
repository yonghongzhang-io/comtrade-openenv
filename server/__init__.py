# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Comtrade Env environment server components."""

__all__ = ["ComtradeEnvironment"]


def __getattr__(name: str):
    if name == "ComtradeEnvironment":
        from .comtrade_env_environment import ComtradeEnvironment

        return ComtradeEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
