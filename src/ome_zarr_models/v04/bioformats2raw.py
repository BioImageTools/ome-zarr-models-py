from typing import Any, Literal

from pydantic import Field

from ome_zarr_models.base import Base
from ome_zarr_models.v04.plate import Plate


class BioFormats2Raw(Base):
    """
    A bioformats2raw zarr group.
    """

    bioformats2raw_layout: Literal[3] = Field(..., alias="bioformats2raw.layout")
    plate: Plate | None = None
    series: Any | None = None
