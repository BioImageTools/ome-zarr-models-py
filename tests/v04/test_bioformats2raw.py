from pathlib import Path

from ome_zarr_models.v04.bioformats2raw import BioFormats2RawAttrs
from ome_zarr_models.v04.plate import (
    Acquisition,
    Column,
    Plate,
    Row,
    WellInPlate,
)


def test_bioformats2raw_exmaple_json():
    with open(Path(__file__).parent / "data" / "bioformats2raw_example.json") as f:
        model = BioFormats2RawAttrs.model_validate_json(f.read())

    assert model == BioFormats2RawAttrs(
        bioformats2raw_layout=3,
        plate=Plate(
            acquisitions=[
                Acquisition(id=0, maximumfieldcount=None, name=None, description=None)
            ],
            columns=[Column(name="1")],
            field_count=1,
            name="Plate Name 0",
            rows=[Row(name="A")],
            version="0.4",
            wells=[WellInPlate(path="A/1", rowIndex=0, columnIndex=0)],
        ),
        series=None,
    )
