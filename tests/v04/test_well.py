import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from ome_zarr_models.v04.well import ImageInWell, Well


def check_against_json(json_path: Path, expected_model: BaseModel) -> None:
    with open(json_path) as f:
        data = json.load(f)

    assert type(expected_model)(**data["well"]) == expected_model


@pytest.mark.parametrize(
    ("filename", "model"),
    [
        (
            "well_example_1.json",
            Well(
                images=[
                    ImageInWell(path="0", acquisition=1),
                    ImageInWell(path="1", acquisition=1),
                    ImageInWell(path="2", acquisition=2),
                    ImageInWell(path="3", acquisition=2),
                ],
                version="0.4",
            ),
        ),
        (
            "well_example_2.json",
            Well(
                images=[
                    ImageInWell(path="0", acquisition=0),
                    ImageInWell(path="1", acquisition=3),
                ],
                version="0.4",
            ),
        ),
    ],
)
def test_examples_valid(filename: str, model: Well):
    with open(Path(__file__).parent / "data" / filename) as f:
        data = json.load(f)

    assert Well(**data["well"]) == model
