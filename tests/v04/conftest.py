from collections.abc import Sequence
from typing import Any, Literal

import numcodecs
import numpy as np
import numpy.typing as npt
from numcodecs.abc import Codec
from pydantic_zarr.v2 import ArraySpec, GroupSpec
from zarr.util import guess_chunks

from ome_zarr_models.v04.axes import Axis
from ome_zarr_models.v04.image import Image, ImageAttrs
from ome_zarr_models.v04.multiscales import (
    Dataset,
    Multiscale,
)


def normalize_chunks(
    chunks: Any,
    shapes: tuple[tuple[int, ...], ...],
    typesizes: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    """
    If chunks is "auto", then use zarr default chunking based on the
    largest array for all the arrays.
    If chunks is a sequence of ints, then use those chunks for all arrays.
    If chunks is a sequence of sequences of ints, then use those chunks for each array.
    """
    if chunks == "auto":
        # sort shapes by descending size
        params_sorted_descending = sorted(
            zip(shapes, typesizes, strict=False),
            key=lambda v: np.prod(v[0]),
            reverse=True,
        )
        return (guess_chunks(*params_sorted_descending[0]),) * len(shapes)
    if isinstance(chunks, Sequence):
        if all(isinstance(element, int) for element in chunks):
            return (tuple(chunks),) * len(shapes)
        if all(isinstance(element, Sequence) for element in chunks):
            if all(all(isinstance(k, int) for k in v) for v in chunks):
                return tuple(map(tuple, chunks))
            else:
                msg = f"Expected a sequence of sequences of ints. Got {chunks} instead."
                raise ValueError(msg)
    msg = f'Input must be a sequence or the string "auto". Got {type(chunks)}'
    raise TypeError(msg)


def from_arrays(
    arrays: Sequence[np.ndarray],
    *,
    paths: Sequence[str],
    axes: Sequence[Axis],
    scales: Sequence[tuple[int | float, ...]],
    translations: Sequence[tuple[int | float, ...]],
    name: str | None = None,
    type: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | tuple[tuple[int, ...], ...] | Literal["auto"] = "auto",
    compressor: Codec | Literal["auto"] = "auto",
    fill_value: Any = 0,
    order: Literal["C", "F", "auto"] = "auto",
) -> Image:
    """
    Create a `Image` from a sequence of multiscale arrays
    and spatial metadata.

    The arrays are used as templates for corresponding `ArraySpec` instances,
    which model the Zarr arrays that would be created if the `Image`
    was stored.

    Parameters
    ----------
    paths: Sequence[str]
        The paths to the arrays.
    axes: Sequence[Axis]
        `Axis` objects describing the dimensions of the arrays.
    arrays: Sequence[ArrayLike] | Sequence[ChunkedArrayLike]
        A sequence of array-like objects that collectively represent the same image
        at multiple levels of detail.
        The attributes of these arrays are used to create `ArraySpec` objects
        that model Zarr arrays stored in the Zarr group.
    scales: Sequence[Sequence[int | float]]
        A scale value for each axis of the array, for each array in `arrays`.
    translations: Sequence[Sequence[int | float]]
        A translation value for each axis the array, for each array in `arrays`.
    name: str | None, default = None
        A name for the multiscale collection. Optional.
    type: str | None, default = None
        A description of the type of multiscale image represented by this group.
        Optional.
    metadata: Dict[str, Any] | None, default = None
        Arbitrary metadata associated with this multiscale collection. Optional.
    chunks: tuple[int] | tuple[tuple[int, ...]] | Literal["auto"], default = "auto"
        The chunks for the arrays in this multiscale group.
        If the string "auto" is provided, each array will have chunks set to the
        zarr-python default value, which depends on the shape and dtype of the array.
        If a single sequence of ints is provided, then this defines the
        chunks for all arrays. If a sequence of sequences of ints is provided,
        then this defines the chunks for each array.
    fill_value: Any, default = 0
        The fill value for the Zarr arrays.
    compressor: `Codec` | "auto", default = `numcodecs.ZStd`
        The compressor to use for the arrays. Default is `numcodecs.ZStd`.
    order: "auto" | "C" | "F"
        The memory layout used for chunks of Zarr arrays.
        The default is "auto", which will infer the order from the input arrays,
        and fall back to "C" if that inference fails.
    """

    chunks_normalized = normalize_chunks(
        chunks,
        shapes=tuple(s.shape for s in arrays),
        typesizes=tuple(s.dtype.itemsize for s in arrays),
    )

    members_flat = {
        "/" + key.lstrip("/"): ArraySpec.from_array(
            array=arr,
            chunks=cnks,
            attributes={},
            compressor=compressor,
            filters=None,
            fill_value=fill_value,
            order=order,
        )
        for key, arr, cnks in zip(paths, arrays, chunks_normalized, strict=False)
    }

    multimeta = Multiscale(
        name=name,
        type=type,
        metadata=metadata,
        axes=tuple(axes),
        datasets=tuple(
            Dataset.build(path=path, scale=scale, translation=translation)
            for path, scale, translation in zip(
                paths, scales, translations, strict=False
            )
        ),
        coordinateTransformations=None,
    )
    return Image(
        members=GroupSpec.from_flat(members_flat).members,
        attributes=ImageAttrs(multiscales=(multimeta,)),
    )


def from_array_props(
    dtype: npt.DTypeLike,
    shapes: Sequence[Sequence[int]],
    paths: Sequence[str],
    axes: Sequence[Axis],
    scales: Sequence[tuple[int | float, ...]],
    translations: Sequence[tuple[int | float, ...]],
    name: str | None = None,
    type: str | None = None,
    metadata: dict[str, Any] | None = None,
    chunks: tuple[int, ...] | tuple[tuple[int, ...], ...] | Literal["auto"] = "auto",
    compressor: Codec | Literal["auto"] = "auto",
    fill_value: Any = 0,
    order: Literal["C", "F"] = "C",
) -> Image:
    """
    Create a `Image` from a dtype and a sequence of shapes.

    The dtype and shapes are used to parametrize `ArraySpec` instances which model the
    Zarr arrays that would be created if the `Image` was stored.

    Parameters
    ----------
    dtype: np.dtype[Any]
        The data type of the arrays.
    shapes: Sequence[Sequence[str]]
        The shapes of the arrays.
    paths: Sequence[str]
        The paths to the arrays.
    axes: Sequence[Axis]
        `Axis` objects describing the dimensions of the arrays.
    scales: Sequence[Sequence[int | float]]
        A scale value for each axis of the array, for each shape in `shapes`.
    translations: Sequence[Sequence[int | float]]
        A translation value for each axis the array, for each shape in `shapes`.
    name: str | None, default = None
        A name for the multiscale collection. Optional.
    type: str | None, default = None
        A description of the type of multiscale image represented by this group.
        Optional.
    metadata: Dict[str, Any] | None, default = None
        Arbitrary metadata associated with this multiscale collection. Optional.
    chunks: tuple[int] | tuple[tuple[int, ...]] | Literal["auto"], default = "auto"
        The chunks for the arrays in this multiscale group.
        If the string "auto" is provided, each array will have chunks set to the
        zarr-python default value, which depends on the shape and dtype of the array.
        If a single sequence of ints is provided, then this defines the chunks for
        all arrays. If a sequence of sequences of ints is provided, then this defines
        the chunks for each array.
    fill_value: Any, default = 0
        The fill value for the Zarr arrays.
    compressor: `Codec`
        The compressor to use for the arrays. Default is `numcodecs.ZStd`.
    order: "C" | "F", default = "C"
        The memory layout used for chunks of Zarr arrays. The default is "C".
    """

    dtype_normalized = np.dtype(dtype)
    if compressor == "auto":
        compressor_parsed = numcodecs.Zstd(level=3)
    else:
        compressor_parsed = compressor
    chunks_normalized = normalize_chunks(
        chunks,
        shapes=tuple(tuple(s) for s in shapes),
        typesizes=tuple(dtype_normalized.itemsize for s in shapes),
    )

    members_flat = {
        "/" + key.lstrip("/"): ArraySpec(
            dtype=dtype,
            shape=shape,
            chunks=cnks,
            attributes={},
            compressor=compressor_parsed,
            filters=None,
            fill_value=fill_value,
            order=order,
        )
        for key, shape, cnks in zip(paths, shapes, chunks_normalized, strict=False)
    }

    multimeta = Multiscale(
        name=name,
        type=type,
        metadata=metadata,
        axes=tuple(axes),
        datasets=tuple(
            Dataset.build(path=path, scale=scale, translation=translation)
            for path, scale, translation in zip(
                paths, scales, translations, strict=False
            )
        ),
        coordinateTransformations=None,
    )
    return Image(
        members=GroupSpec.from_flat(members_flat).members,
        attributes=ImageAttrs(multiscales=(multimeta,)),
    )
