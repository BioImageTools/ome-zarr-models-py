"""
For reference, see the [coordinate transformations section of the OME-zarr specification](https://ngff.openmicroscopy.org/0.4/#trafo-md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field
from typing_extensions import Self

from ome_zarr_models.base import Base

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "Identity",
    "PathScale",
    "PathTranslation",
    "ScaleTransform",
    "TranslationTransform",
    "VectorScale",
    "VectorTransform",
    "VectorTranslation",
]


class Transformation(Base):
    """
    Base transformation class.
    """

    type: Literal["identity", "translation", "scale"]


class Identity(Transformation):
    """
    Identity transformation.

    Notes
    -----
    Although defined in the specification, it is not allowed
    to be used anywhere.
    """

    type: Literal["identity"]


class VectorScale(Transformation):
    """
    Scale transformation parametrized by a vector of numbers.
    """

    type: Literal["scale"]
    scale: list[float]

    @classmethod
    def build(cls, data: Iterable[float]) -> Self:
        """
        Create a VectorScale from an iterable of floats.
        """
        return cls(type="scale", scale=list(data))

    @property
    def ndim(self) -> int:
        """
        Number of dimensions.
        """
        return len(self.scale)


class PathScale(Transformation):
    """
    Scale transformation parametrized by a path.
    """

    type: Literal["scale"]
    path: str


class VectorTranslation(Transformation):
    """
    Translation transformation parametrized by a vector of numbers.
    """

    type: Literal["translation"] = Field(..., description="Type")
    translation: list[float]

    @classmethod
    def build(cls, data: Iterable[float]) -> Self:
        """
        Create a VectorTranslation from an iterable of floats.
        """
        return cls(type="translation", translation=list(data))

    @property
    def ndim(self) -> int:
        """
        Number of dimensions.
        """
        return len(self.translation)


class PathTranslation(Transformation):
    """
    Translation transformation parametrized by a path.
    """

    type: Literal["translation"]
    translation: str


ScaleTransform = VectorScale | PathScale
TranslationTransform = VectorTranslation | PathTranslation
VectorTransform = VectorScale | VectorTranslation


def _ndim(transform: VectorTransform) -> int:
    """
    Get the dimensionality of a scale or translation transform.
    """
    return transform.ndim


def _build_transforms(
    scale: Sequence[float], translation: Sequence[float] | None
) -> tuple[VectorScale] | tuple[VectorScale, VectorTranslation]:
    """
    Create a `VectorScale` and optionally a `VectorTranslation` from a scale and a
    translation parameter.
    """
    vec_scale = VectorScale.build(scale)
    if translation is None:
        return (vec_scale,)
    else:
        vec_trans = VectorTranslation.build(translation)
        return vec_scale, vec_trans
