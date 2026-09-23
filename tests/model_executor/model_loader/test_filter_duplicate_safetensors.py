# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
)
from vllm.model_executor.models.iquest_q1_mtp import IQuestQ1MTP


def test_filter_duplicate_safetensors_files_missing_weight():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_file = os.path.join(tmpdir, "model-00001-of-00002.safetensors")
        with open(existing_file, "wb") as f:
            f.write(b"")

        existing_file2 = os.path.join(tmpdir, "model-00002-of-00002.safetensors")
        with open(existing_file2, "wb") as f:
            f.write(b"")

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
                "layer.2.weight": "model-00003-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        hf_weights_files = [
            os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
        ]

        with pytest.raises(FileNotFoundError) as exc_info:
            filter_duplicate_safetensors_files(
                hf_weights_files=hf_weights_files,
                hf_folder=tmpdir,
                index_file="model.safetensors.index.json",
            )

        assert "model-00003-of-00002.safetensors" in str(exc_info.value)


def test_filter_duplicate_safetensors_files_all_exist():
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_files = []
        for i in range(1, 3):
            file_path = os.path.join(tmpdir, f"model-0000{i}-of-00002.safetensors")
            with open(file_path, "wb") as f:
                f.write(b"")
            existing_files.append(file_path)

        index_file = os.path.join(tmpdir, "model.safetensors.index.json")
        index_content = {
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors",
            }
        }
        with open(index_file, "w") as f:
            json.dump(index_content, f)

        filter_duplicate_safetensors_files(
            hf_weights_files=existing_files,
            hf_folder=tmpdir,
            index_file="model.safetensors.index.json",
        )


@pytest.mark.parametrize("mode", ["mtp", "no_filter", "no_index", "no_match"])
def test_loader_prunes_only_unneeded_mtp_shards(tmp_path, mode):
    """Exercise the model predicate through the actual file iterator."""
    shared = {
        "model.embed_tokens.weight": torch.ones(2, 2),
        "lm_head.weight": torch.full((2, 2), 2.0),
        "model.layers.0.weight": torch.full((2, 2), 3.0),
    }
    draft = {"mtp_layers.0.eh_proj.weight": torch.full((2, 2), 4.0)}
    backbone = {"model.layers.1.weight": torch.full((2, 2), 5.0)}
    shards = {
        "shared.safetensors": shared,
        "mtp.safetensors": draft,
        "backbone.safetensors": backbone,
    }
    weight_map = {}
    for filename, weights in shards.items():
        save_file(weights, str(tmp_path / filename))
        weight_map.update({name: filename for name in weights})
    if mode != "no_index":
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": weight_map})
        )
    model = torch.nn.Module()
    if mode != "no_filter":
        model.safetensors_weights_filter = (
            IQuestQ1MTP.safetensors_weights_filter
            if mode != "no_match"
            else lambda name: False
        )
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    loaded = dict(
        loader.get_all_weights(
            SimpleNamespace(model=str(tmp_path), revision=None), model
        )
    )
    expected = shared | draft
    if mode != "mtp":
        expected |= backbone
    assert loaded.keys() == expected.keys()
    for name, weight in expected.items():
        torch.testing.assert_close(loaded[name], weight, rtol=0, atol=0)


if __name__ == "__main__":
    test_filter_duplicate_safetensors_files_missing_weight()
    test_filter_duplicate_safetensors_files_all_exist()
