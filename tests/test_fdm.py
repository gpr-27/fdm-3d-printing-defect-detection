"""CPU tests (no dataset or downloads needed). Run from the repo root:  PYTHONPATH=src pytest -q"""
import os

import pytest
from PIL import Image

from fdm_defects.data import grouped_split, load_folder, random_split_80_20, session_key
from fdm_defects.models import build_model, count_trainable


@pytest.fixture()
def fake_folder(tmp_path):
    """Two classes, two capture sessions each, 5 frames per session."""
    for cls, sessions in {"Cracking": ["20231128", "20240118"], "Warping": ["20231228", "20240102"]}.items():
        os.makedirs(tmp_path / cls)
        for s in sessions:
            for k in range(5):
                Image.new("RGB", (32, 24), (k * 40, 10, 10)).save(tmp_path / cls / f"Image_{s}1953{k:02d}980.jpg")
    return str(tmp_path)


def test_random_split_sizes_and_determinism(fake_folder):
    ds = load_folder(fake_folder, 32)
    a_tr, a_va = random_split_80_20(ds, seed=42)
    b_tr, b_va = random_split_80_20(ds, seed=42)
    assert (len(a_tr), len(a_va)) == (16, 4)            # int(0.8 * 20) = 16
    assert list(a_va.indices) == list(b_va.indices)


def test_grouped_split_keeps_sessions_together(fake_folder):
    ds = load_folder(fake_folder, 32)
    tr, va = grouped_split(ds, 0.2, seed=0)
    key = lambda i: session_key(ds.samples[i][0], ds.classes[ds.samples[i][1]])
    assert not ({key(i) for i in tr.indices} & {key(i) for i in va.indices})
    assert len(tr) + len(va) == len(ds) and len(va) > 0


def test_session_key_formats():
    assert session_key("x/Image_20231128195336980.jpg", "Cracking") == "Cracking@2023-11-28"
    assert session_key("x/scratch_2_17.jpg", "no_defected") == "no_defected@scratch_2"


@pytest.mark.parametrize("name,classes,expected", [
    ("alexnet", 5, 54_554_629),        # value printed in 1_alexnet_fdm3d_defects.ipynb
    ("resnet18", 5, 2_565),            # value printed when 2_resnet_fdm3d_defects.ipynb was run
    ("inception_v3", 5, 2_571_786),    # value printed in 3_inception_fdm3d_defects.ipynb
    ("resnet50", 5, 10_245),
    ("alexnet", 2, 54_542_338),
    ("resnet18", 2, 1_026),
])
def test_trainable_parameter_counts(name, classes, expected):
    assert count_trainable(build_model(name, classes, pretrained=False)) == expected
