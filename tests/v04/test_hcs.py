from pathlib import Path

import zarr

from ome_zarr_models.v04 import HCS
from ome_zarr_models.v04.hcs import HCSAttrs
from ome_zarr_models.v04.plate import Acquisition, Column, Plate, Row, WellInPlate
from ome_zarr_models.v04.well import Well, WellImage


def test_example_hcs() -> None:
    group = zarr.open(Path(__file__).parent / "data" / "hcs_example.ome.zarr", mode="r")
    hcs = HCS.from_zarr(group)
    assert hcs.attributes == HCSAttrs(
        plate=Plate(
            acquisitions=[
                Acquisition(
                    id=0,
                    name="20200812-CardiomyocyteDifferentiation14-Cycle1",
                    maximumfieldcount=None,
                    description=None,
                    starttime=None,
                    endtime=None,
                )
            ],
            columns=[Column(name="03")],
            field_count=None,
            name=None,
            rows=[Row(name="B")],
            version="0.4",
            wells=[WellInPlate(path="B/03", rowIndex=0, columnIndex=0)],
        )
    )

    well_groups = list(hcs.well_groups)
    assert len(well_groups) == 1
    assert well_groups[0].attributes.well == Well(
        images=[WellImage(path="0", acquisition=None)], version="0.4"
    )
