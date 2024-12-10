"""
For reference, see the [multiscales section of the OME-zarr specification](https://ngff.openmicroscopy.org/0.4/#multiscale-md).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Annotated, Any, Self, get_args

from pydantic import AfterValidator, Field, model_validator

from ome_zarr_models._utils import duplicates
from ome_zarr_models.base import Base
from ome_zarr_models.v04.axes import Axes, AxisType
from ome_zarr_models.v04.coordinate_transformations import (
    ScaleTransform,
    TranslationTransform,
    VectorTransform,
    _build_transforms,
    _ndim,
)

__all__ = ["Dataset", "Multiscale", "Multiscales"]

VALID_NDIM = (2, 3, 4, 5)
ValidTransform = tuple[ScaleTransform] | tuple[ScaleTransform, TranslationTransform]


def _ensure_transform_dimensionality(
    transforms: ValidTransform,
) -> ValidTransform:
    """
    Ensures that the elements in the input sequence define transformations with
    identical dimensionality. If any of the transforms are defined with a path
    instead of concrete values, then no validation will be performed and the
    transforms will be returned as-is.
    """
    vector_transforms = filter(lambda v: isinstance(v, VectorTransform), transforms)
    ndims = tuple(map(_ndim, vector_transforms))  # type: ignore[arg-type]
    ndims_set = set(ndims)
    if len(ndims_set) > 1:
        msg = (
            "The transforms have inconsistent dimensionality. "
            f"Got transforms with dimensionality = {ndims}."
        )
        raise ValueError(msg)
    return transforms


def _ensure_scale_translation(
    transforms: ValidTransform,
) -> ValidTransform:
    """
    Ensures that
    - there are only 1 or 2 transforms.
    - the first element is a scale transformation
    - the second element, if present, is a translation transform
    """
    if len(transforms) not in (1, 2):
        msg = f"Invalid number of transforms: got {len(transforms)}, expected 1 or 2"
        raise ValueError(msg)

    maybe_scale = transforms[0]
    if maybe_scale.type != "scale":
        msg = (
            "The first element of `coordinateTransformations` must be a scale "
            f"transform. Got {maybe_scale} instead."
        )
        raise ValueError(msg)
    if len(transforms) == 2:
        maybe_trans = transforms[1]
        if (maybe_trans.type) != "translation":
            msg = (
                "The second element of `coordinateTransformations` must be a "
                f"translation transform. Got {maybe_trans} instead."
            )
            raise ValueError(msg)

    return transforms


def _ensure_axis_length(axes: Axes) -> Axes:
    """
    Ensures that there are between 2 and 5 axes (inclusive)
    """
    if (len_axes := len(axes)) not in VALID_NDIM:
        msg = (
            f"Incorrect number of axes provided ({len_axes}). "
            "Only 2, 3, 4, or 5 axes are allowed."
        )
        raise ValueError(msg)
    return axes


def _ensure_unique_axis_names(axes: Axes) -> Axes:
    """
    Ensures that the names of the axes are unique.
    """
    name_dupes = duplicates(a.name for a in axes)
    if len(name_dupes) > 0:
        msg = (
            f"Axis names must be unique. Axis names {tuple(name_dupes.keys())} are "
            "repeated."
        )
        raise ValueError(msg)
    return axes


def _ensure_axis_types(axes: Axes) -> Axes:
    """
    Ensures that the following conditions are true:

    - there are only 2 or 3 axes with type `space`
    - the axes with type `space` are last in the list of axes
    - there is only 1 axis with type `time`
    - there is only 1 axis with type `channel`
    - there is only 1 axis with a type that is not `space`, `time`, or `channel`
    """
    axis_types = [ax.type for ax in axes]
    type_census = Counter(axis_types)
    num_spaces = type_census["space"]
    if num_spaces not in [2, 3]:
        msg = (
            f"Invalid number of space axes: {num_spaces}. "
            "Only 2 or 3 space axes are allowed."
        )
        raise ValueError(msg)

    if not all(a == "space" for a in axis_types[-num_spaces:]):
        msg = f"Space axes must come last. Got axes with order: {axis_types}."
        raise ValueError(msg)

    if (num_times := type_census["time"]) > 1:
        msg = f"Invalid number of time axes: {num_times}. Only 1 time axis is allowed."
        raise ValueError(msg)

    if (num_channels := type_census["channel"]) > 1:
        msg = (
            f"Invalid number of channel axes: {num_channels}. "
            "Only 1 channel axis is allowed."
        )
        raise ValueError(msg)

    custom_axes = set(axis_types) - set(get_args(AxisType))
    if (num_custom := len(custom_axes)) > 1:
        msg = (
            f"Invalid number of custom axes: {num_custom}. "
            "Only 1 custom axis is allowed."
        )
        raise ValueError(msg)
    return axes


class Dataset(Base):
    """
    An element of Multiscale.datasets.
    """

    # TODO: validate that path resolves to an actual zarr array
    # TODO: can we validate that the paths must be ordered from highest resolution to
    # smallest using scale metadata?
    path: str
    coordinateTransformations: Annotated[
        ValidTransform,
        AfterValidator(_ensure_scale_translation),
        AfterValidator(_ensure_transform_dimensionality),
    ]

    @classmethod
    def build(
        cls, *, path: str, scale: Sequence[float], translation: Sequence[float]
    ) -> Self:
        """
        Construct a `Dataset` from a path, a scale, and a translation.
        """
        return cls(
            path=path,
            coordinateTransformations=_build_transforms(
                scale=scale, translation=translation
            ),
        )


Datasets = Sequence[Dataset]


class Multiscale(Base):
    """
    An element of multiscales metadata.
    """

    axes: Annotated[
        Axes,
        AfterValidator(_ensure_axis_length),
        AfterValidator(_ensure_unique_axis_names),
        AfterValidator(_ensure_axis_types),
    ]
    datasets: Datasets = Field(..., min_length=1)
    version: Any | None = None
    coordinateTransformations: ValidTransform | None = None
    metadata: Any = None
    name: Any | None = None
    type: Any = None

    @property
    def ndim(self) -> int:
        """
        Dimensionality of the data described by this metadata.

        Determined by the length of the axes attribute.
        """
        return len(self.axes)

    @model_validator(mode="after")
    def _ensure_top_transforms_dimensionality(self) -> Self:
        """
        Ensure that the dimensionality of the top-level coordinateTransformations,
        if present, is consistent with the rest of the model.
        """
        ctx = self.coordinateTransformations
        if ctx is not None:
            # check that the dimensionality of the coordinateTransformations is
            # internally consistent
            _ = _ensure_transform_dimensionality(ctx)

        return self

    @model_validator(mode="after")
    def _ensure_axes_top_transforms(data: Multiscale) -> Multiscale:
        """
        Ensure that the length of the axes matches the dimensionality of the transforms
        defined in the top-level coordinateTransformations, if present.
        """
        self_ndim = len(data.axes)
        if data.coordinateTransformations is not None:
            for tx in data.coordinateTransformations:
                if hasattr(tx, "ndim") and self_ndim != tx.ndim:
                    msg = (
                        f"The length of axes does not match the dimensionality of "
                        f"the {tx.type} transform in coordinateTransformations. "
                        f"Got {self_ndim} axes, but the {tx.type} transform has "
                        f"dimensionality {tx.ndim}"
                    )
                    raise ValueError(msg)
        return data

    @model_validator(mode="after")
    def _ensure_axes_dataset_transforms(data: Multiscale) -> Multiscale:
        """
        Ensure that the length of the axes matches the dimensionality of the transforms
        """
        self_ndim = len(data.axes)
        for ds_idx, ds in enumerate(data.datasets):
            for tx in ds.coordinateTransformations:
                if hasattr(tx, "ndim") and self_ndim != tx.ndim:
                    msg = (
                        f"The length of axes does not match the dimensionality of "
                        f"the {tx.type} transform in "
                        f"datasets[{ds_idx}].coordinateTransformations. "
                        f"Got {self_ndim} axes, but the {tx.type} transform has "
                        f"dimensionality {tx.ndim}"
                    )
                    raise ValueError(msg)
        return data


Multiscales = Sequence[Multiscale]
