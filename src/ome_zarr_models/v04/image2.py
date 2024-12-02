from typing import Self

import zarr
from pydantic import BaseModel, ConfigDict, model_validator

from ome_zarr_models.v04.image import ImageAttrs


class Image2(BaseModel):
    """
    A OME-zarr image.

    This is represented by a single zarr Group. To get the OME-zarr
    metadata, use the `.attributes` property.

    To modify the attributes, get a copy of them using the `.attributes`
    property, modify them, and then set the attributes property with
    the modified version.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    group: zarr.Group

    @property
    def attributes(self) -> ImageAttrs:
        print(self.group.attrs.asdict())
        return ImageAttrs(**self.group.attrs.asdict())

    @attributes.setter
    def attributes(self, attrs: ImageAttrs) -> None:
        self.group.attrs.put(attrs.model_dump(exclude_none=True))

    @model_validator(mode="after")
    def _check_multiscale_arrays(self) -> Self:
        """
        Check that all the arrays referenced by the `multiscales` metadata meet the
        following criteria:

            - they exist
            - they are not groups
            - they have dimensionality consistent with the number of axes defined in the
              metadata.
        """
        multiscales = self.attributes.multiscales

        for multiscale in multiscales:
            for dataset in multiscale.datasets:
                if dataset.path not in self.group:
                    msg = (
                        f"The multiscale metadata references an array that does not "
                        f"exist in this group: {dataset.path}"
                    )
                    raise ValueError(msg)

                arr = self.group[dataset.path]
                if isinstance(arr, zarr.Group):
                    msg = f"The node at {dataset.path} is a group, not an array."
                    raise ValueError(msg)

                if arr.ndim != multiscale.ndim:
                    msg = (
                        f"The multiscale metadata has {multiscale.ndim} axes "
                        "which does not match the dimensionality of the array "
                        f"found in this group at {dataset.path} ({arr.ndim}). "
                        "The number of axes must match the array dimensionality."
                    )
                    raise ValueError(msg)
        return self
