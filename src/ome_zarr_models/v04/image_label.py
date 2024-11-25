from __future__ import annotations

import warnings
from collections import Counter
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import AfterValidator, Field, model_validator

from ome_zarr_models.base import Base
from ome_zarr_models.v04.image import ImageAttrs

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable

__all__ = ["ConInt", "RGBA", "Color", "Source", "Property", "ImageLabel", "GroupAttrs"]

ConInt = Annotated[int, Field(strict=True, ge=0, le=255)]
RGBA = tuple[ConInt, ConInt, ConInt, ConInt]


def _duplicates(values: Iterable[Hashable]) -> dict[Hashable, int]:
    """
    Takes a sequence of hashable elements and returns a dict where the keys are the
    elements of the input that occurred at least once, and the values are the
    frequencies of those elements.
    """
    counts = Counter(values)
    return {k: v for k, v in counts.items() if v > 1}


class Color(Base):
    """
    A label value and RGBA.

    References
    ----------
    https://ngff.openmicroscopy.org/0.4/#label-md
    """

    label_value: int = Field(..., serialization_alias="label-value")
    rgba: RGBA | None


class Source(Base):
    """
    Source data for the labels.
    """

    # TODO: add validation that this path resolves to something
    image: str | None = "../../"


class Property(Base):
    """
    A single property.
    """

    label_value: int = Field(..., serialization_alias="label-value")


def _parse_colors(colors: list[Color] | None) -> list[Color] | None:
    if colors is None:
        msg = (
            "The field `colors` is `None`. Version 0.4 of"
            "the OME-NGFF spec states that `colors` should be a list of "
            "label descriptors."
        )
        warnings.warn(msg, stacklevel=1)
    else:
        dupes = _duplicates(x.label_value for x in colors)
        if len(dupes) > 0:
            msg = (
                f"Duplicated label-value: {tuple(dupes.keys())}."
                "label-values must be unique across elements of `colors`."
            )
            raise ValueError(msg)

    return colors


def _parse_version(version: Literal["0.4"] | None) -> Literal["0.4"] | None:
    if version is None:
        _ = (
            "The `version` attribute is `None`. Version 0.4 of "
            "the OME-NGFF spec states that `version` should either be unset or "
            "the string 0.4"
        )
    return version


def _parse_imagelabel(model: ImageLabel) -> ImageLabel:
    """
    check that label_values are consistent across properties and colors
    """
    if model.colors is not None and model.properties is not None:
        prop_label_value = [prop.label_value for prop in model.properties]
        color_label_value = [color.label_value for color in model.colors]

        prop_label_value_set = set(prop_label_value)
        color_label_value_set = set(color_label_value)
        if color_label_value_set != prop_label_value_set:
            msg = (
                "Inconsistent `label_value` attributes in `colors` and `properties`."
                f"The `properties` attributes have `label_values` {prop_label_value}, "
                f"The `colors` attributes have `label_values` {color_label_value}, "
            )
            raise ValueError(msg)
    return model


class ImageLabel(Base):
    """
    image-label metadata.
    See https://ngff.openmicroscopy.org/0.4/#label-md
    """

    _version: Literal["0.4"]

    version: Annotated[Literal["0.4"] | None, AfterValidator(_parse_version)]
    colors: Annotated[tuple[Color, ...] | None, AfterValidator(_parse_colors)] = None
    properties: tuple[Property, ...] | None = None
    source: Source | None = None

    @model_validator(mode="after")
    def _parse_model(self) -> ImageLabel:
        return _parse_imagelabel(self)


class GroupAttrs(ImageAttrs):
    """
    Attributes for a Zarr group that contains `image-label` metadata.
    Inherits from `v04.multiscales.MultiscaleAttrs`.

    See https://ngff.openmicroscopy.org/0.4/#label-md

    Attributes
    ----------
    image_label: `ImageLabel`
        Image label metadata.
    multiscales: tuple[v04.multiscales.Multiscales]
        Multiscale image metadata.
    """

    image_label: Annotated[ImageLabel, Field(..., serialization_alias="image-label")]
