import json
from pathlib import Path

from ome_zarr_models.v04.bioformats2raw import BioFormats2Raw
from ome_zarr_models.v04.plate import (
    AcquisitionInPlate,
    ColumnInPlate,
    Plate,
    RowInPlate,
    WellInPlate,
)


def test_bioformats2raw_exmaple_json():
    with open(Path(__file__).parent / "data" / "bioformats2raw_example.json") as f:
        data = json.load(f)

    assert BioFormats2Raw(**data["bioformats"])
