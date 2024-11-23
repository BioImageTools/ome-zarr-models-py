import pydantic


class Base(pydantic.BaseModel):
    """
    The base pydantic model for all metadata classes
    """

    class Config:
        """
        Pydantic config.
        """

        validate_assignment = True
        extra = "allow"
        # This allows fields with aliases to be populated by either
        # their alias or class attribute name
        #
        # We use this so we can handle (at least) the "bioformats2raw.version"
        # key - attributes in Python can't contain a "."
        populate_by_name = True
