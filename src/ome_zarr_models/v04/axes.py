from collections.abc import Sequence
from typing import Any, Literal

from ome_zarr_models.base import Base

__all__ = ["AxisType", "Axis", "Axes"]


AxisType = Literal["space", "time", "channel"]


class Axis(Base):
    """
    Model for an element of `Multiscale.axes`.

    See https://ngff.openmicroscopy.org/0.4/#axes-md.
    """

    name: str
    type: str | None = None
    unit: str | Any | None = None


Axes = Sequence[Axis]
