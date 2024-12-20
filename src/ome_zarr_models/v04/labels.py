from __future__ import annotations

from pydantic import Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from ome_zarr_models.v04.base import Base

# Labels is imported to the `ome_zarr_py.v04` namespace, so not
# listed here
__all__ = ["LabelsAttrs"]


class LabelsAttrs(Base):
    """
    Attributes for an OME-Zarr labels dataset.
    """

    labels: list[str] = Field(
        ..., description="List of paths to labels arrays within a labels dataset."
    )


class Labels(GroupSpec[LabelsAttrs, ArraySpec | GroupSpec]):
    """
    An OME-zarr labels dataset.
    """
