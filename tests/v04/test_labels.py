import zarr

from ome_zarr_models.v04.image import Image
from ome_zarr_models.v04.labels import Labels, LabelsAttrs


def test_labels() -> None:
    # TODO: turn this into a local test
    group = zarr.open_group(
        "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr", mode="r"
    )
    image = Image.from_zarr(group)
    assert image.labels == Labels(
        zarr_version=2, attributes=LabelsAttrs(labels=["0"]), members={}
    )
