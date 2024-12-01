"""
For reference, see the [plate section of the OME-zarr specification](https://ngff.openmicroscopy.org/0.4/index.html#plate-md).
"""

from collections import Counter
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

from ome_zarr_models.base import Base
from ome_zarr_models.utils import _AlphaNumericConstraint, _unique_items_validator

__all__ = [
    "Acquisition",
    "Column",
    "Plate",
    "Row",
    "WellInPlate",
]


class Acquisition(Base):
    """
    A single acquisition.
    """

    id: NonNegativeInt = Field(description="A unique identifier.")
    name: str | None = None
    maximumfieldcount: PositiveInt | None = Field(
        default=None,
        description="Maximum number of fields of view for the acquisition",
    )
    description: str | None = None
    starttime: int | None = None
    endtime: int | None = None


class WellInPlate(Base):
    """
    A single well within a plate.
    """

    # TODO: validate
    # path must be "{name in rows}/{name in columns}"
    path: str
    rowIndex: int
    columnIndex: int


class Column(Base):
    """
    A single column within a well.
    """

    name: Annotated[str, _AlphaNumericConstraint]


class Row(Base):
    """
    A single row within a well.
    """

    name: Annotated[str, _AlphaNumericConstraint]


class Plate(Base):
    """
    A single plate.
    """

    acquisitions: list[Acquisition] | None = None
    columns: Annotated[list[Column], AfterValidator(_unique_items_validator)]
    field_count: PositiveInt | None = Field(
        default=None, description="Maximum number of fields per view across wells"
    )
    name: str | None = Field(default=None, description="Plate name")
    rows: Annotated[list[Row], AfterValidator(_unique_items_validator)]
    # version will become required in 0.5
    version: str | None = Field(None, description="Version of the plate specification")
    wells: list[WellInPlate]

    @model_validator(mode="after")
    def _check_well_paths(self) -> Self:
        """
        Check well paths are valid.
        """
        errors = []
        row_names = {row.name for row in self.rows}
        column_names = {column.name for column in self.columns}

        for well in self.wells:
            path = well.path
            if Counter(path)["/"] != 1:
                errors.append(f"well path '{path}' does not contain a single '/'")
                continue

            row, column = path.split("/")
            if row not in row_names:
                errors.append(
                    f"row '{row}' in well path '{path}' is not in list of rows"
                )
            if column not in column_names:
                errors.append(
                    f"column '{column}' in well path '{path}' is not in list of columns"
                )

        if len(errors) > 0:
            errors_joined = "\n".join(errors)
            raise ValueError(f"Error validating plate metadata:\n{errors_joined}")

        return self
