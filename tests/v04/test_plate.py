import re

import pytest
from pydantic import ValidationError
from tests.v04.conftest import read_in_json

from ome_zarr_models.v04.plate import Acquisition, Column, Plate, Row, WellInPlate


def test_example_plate_json() -> None:
    plate = read_in_json(json_fname="plate_example_1.json", model_cls=Plate)
    assert plate == Plate(
        acquisitions=[
            Acquisition(
                id=1,
                name="Meas_01(2012-07-31_10-41-12)",
                maximumfieldcount=2,
                description=None,
                starttime=1343731272000,
                endtime=None,
            ),
            Acquisition(
                id=2,
                name="Meas_02(201207-31_11-56-41)",
                maximumfieldcount=2,
                description=None,
                starttime=1343735801000,
                endtime=None,
            ),
        ],
        columns=[Column(name="1"), Column(name="2"), Column(name="3")],
        field_count=4,
        name="test",
        rows=[Row(name="A"), Row(name="B")],
        version="0.4",
        wells=[
            WellInPlate(path="A/1", rowIndex=0, columnIndex=0),
            WellInPlate(path="A/2", rowIndex=0, columnIndex=1),
            WellInPlate(path="A/3", rowIndex=0, columnIndex=2),
            WellInPlate(path="B/1", rowIndex=1, columnIndex=0),
            WellInPlate(path="B/2", rowIndex=1, columnIndex=1),
            WellInPlate(path="B/3", rowIndex=1, columnIndex=2),
        ],
    )


def test_example_plate_json_2() -> None:
    plate = read_in_json(json_fname="plate_example_2.json", model_cls=Plate)
    assert plate == Plate(
        acquisitions=[
            Acquisition(
                id=1,
                name="single acquisition",
                maximumfieldcount=1,
                description=None,
                starttime=1343731272000,
                endtime=None,
            )
        ],
        columns=[
            Column(name="1"),
            Column(name="2"),
            Column(name="3"),
            Column(name="4"),
            Column(name="5"),
            Column(name="6"),
            Column(name="7"),
            Column(name="8"),
            Column(name="9"),
            Column(name="10"),
            Column(name="11"),
            Column(name="12"),
        ],
        field_count=1,
        name="sparse test",
        rows=[
            Row(name="A"),
            Row(name="B"),
            Row(name="C"),
            Row(name="D"),
            Row(name="E"),
            Row(name="F"),
            Row(name="G"),
            Row(name="H"),
        ],
        version="0.4",
        wells=[
            WellInPlate(path="C/5", rowIndex=2, columnIndex=4),
            WellInPlate(path="D/7", rowIndex=3, columnIndex=6),
        ],
    )


def test_unique_column_names() -> None:
    with pytest.raises(ValidationError, match="Duplicate values found in"):
        Plate(
            columns=[Column(name="col1"), Column(name="col1")],
            rows=[Row(name="row1")],
            version="0.4",
            wells=[WellInPlate(path="path1", rowIndex=1, columnIndex=1)],
        )


def test_unique_row_names() -> None:
    with pytest.raises(ValidationError, match="Duplicate values found in"):
        Plate(
            columns=[Column(name="col1")],
            rows=[Row(name="row1"), Row(name="row1")],
            version="0.4",
            wells=[WellInPlate(path="path1", rowIndex=1, columnIndex=1)],
        )


@pytest.mark.parametrize("cls", [Row, Column])
def test_alphanumeric_column_names(cls) -> None:
    with pytest.raises(ValidationError, match="String should match pattern "):
        cls(name="col-1")


@pytest.mark.parametrize(
    ("well_path", "msg"),
    [
        ("path1", "well path 'path1' does not contain a single '/'"),
        ("row1/col1/", "well path 'row1/col1/' does not contain a single '/'"),
        (
            "row1/col2",
            "column 'col2' in well path 'row1/col2' is not in list of columns",
        ),
        (
            "row2/col1",
            "row 'row2' in well path 'row2/col1' is not in list of rows",
        ),
    ],
)
def test_well_paths(well_path: str, msg: str) -> None:
    # No separator
    with pytest.raises(ValidationError, match=re.escape(msg)):
        Plate(
            columns=[Column(name="col1")],
            rows=[Row(name="row1")],
            version="0.4",
            wells=[WellInPlate(path=well_path, rowIndex=1, columnIndex=1)],
        )
