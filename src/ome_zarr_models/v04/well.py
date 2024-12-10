"""
For reference, see the [well section of the OME-zarr specification](https://ngff.openmicroscopy.org/0.4/#well-md).
"""

from collections import defaultdict
from collections.abc import Generator
from typing import Annotated, Literal

from pydantic import AfterValidator, Field
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from ome_zarr_models._utils import _AlphaNumericConstraint, _unique_items_validator
from ome_zarr_models.base import Base
from ome_zarr_models.v04.image import Image

# WellGroup is defined one level higher
__all__ = ["Well", "WellAttrs", "WellImage"]


class WellImage(Base):
    """
    A single image within a well.
    """

    path: Annotated[str, _AlphaNumericConstraint]
    acquisition: int | None = Field(
        None, description="A unique identifier within the context of the plate"
    )


class Well(Base):
    """
    A single well
    """

    images: Annotated[list[WellImage], AfterValidator(_unique_items_validator)]
    version: Literal["0.4"] | None = Field(
        None, description="Version of the well specification"
    )

    def get_acquisition_paths(self) -> dict[int, list[str]]:
        """
        Get mapping from acquisition indices to corresponding paths.

        Returns
        -------
        dict
            Dictionary with `(acquisition index: [image_path])` key/value
            pairs.

        Raises
        ------
        ValueError
            If an element of `self.well.images` has no `acquisition` attribute.
        """
        acquisition_dict: dict[int, list[str]] = defaultdict(list)
        for image in self.images:
            if image.acquisition is None:
                raise ValueError(
                    "Cannot get acquisition paths for Zarr files without "
                    "'acquisition' metadata at the well level"
                )
            acquisition_dict[image.acquisition].append(image.path)
        return dict(acquisition_dict)


class WellAttrs(Base):
    """
    Attributes for a well group.
    """

    well: Well


class WellGroup(GroupSpec[WellAttrs, ArraySpec | GroupSpec]):
    def get_image(self, i: int) -> Image:
        """
        Get a single image from this well.
        """
        image = self.attributes.well.images[i]
        image_path = image.path
        image_path_parts = image_path.split("/")
        group = self
        for part in image_path_parts:
            group = group.members[part]

        return Image(attributes=group.attributes, members=group.members)

    @property
    def n_images(self) -> int:
        """
        Number of images.
        """
        return len(self.attributes.well.images)

    @property
    def images(self) -> Generator[Image, None, None]:
        """
        Generator for all images in this well.
        """
        for i in range(self.n_images):
            yield self.get_image(i)
