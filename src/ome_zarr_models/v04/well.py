from typing import Annotated

from pydantic import AfterValidator, Field

from ome_zarr_models.base import Base
from ome_zarr_models.utils import _AlphaNumericConstraint, _unique_items_validator

__all__ = ["ImageInWell", "Well"]


class ImageInWell(Base):
    """
    Model for an element of `Well.images`.

    Notes
    -----
    The NGFF image is defined in a different model
    (`NgffImageMeta`), while the `Image` model only refers to an item of
    `Well.images`.

    References
    ----------
    https://ngff.openmicroscopy.org/0.4/#well-md
    """

    path: Annotated[str, _AlphaNumericConstraint]
    acquisition: int | None = Field(
        None, description="A unique identifier within the context of the plate"
    )


class Well(Base):
    """
    Model for `NgffWellMeta.well`.

    References
    ----------
    https://ngff.openmicroscopy.org/0.4/#well-md
    """

    images: Annotated[list[ImageInWell], AfterValidator(_unique_items_validator)]
    version: str | None = Field(None, description="Version of the well specification")
