from pydantic import BaseModel, ConfigDict


class Base(BaseModel):
    """
    The base pydantic model for all metadata classes
    """

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
        # This allows fields with aliases to be populated by either
        # their alias or class attribute name
        #
        # We use this so we can handle e.g., the "bioformats2raw.version"
        # key - names in Python can't contain a "."
        populate_by_name=True,
        frozen=True,
    )
