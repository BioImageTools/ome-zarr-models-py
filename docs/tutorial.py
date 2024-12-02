# # Tutorial

import zarr
import zarr.storage
from pydantic import ValidationError
from rich.pretty import pprint

from ome_zarr_models.data import tutorial_data_path
from ome_zarr_models.v04 import Image2
from ome_zarr_models.v04.coordinate_transformations import (
    VectorTranslation,
)

# ## Creating models
#
# We can create an Image model from a zarr group, that points to an
# OME-zarr dataset:

print(tutorial_data_path.absolute())
group = zarr.open(tutorial_data_path)
ome_zarr_image = Image2(group=group)
pprint(ome_zarr_image)

# This image contains only the zarr group. To get a copy of the attributes,
# we can access the `.attributes` property

metadata = ome_zarr_image.attributes
pprint(metadata)

# Note that this is a copy of the metadata, and editing it will not automatically
# update the attributes stored in the zarr group. We will see how to do this later.

# ## Updating models
#
# All the fields in the models can be updated in place. When you do this, any
# validation on the individual field you are updating will take place.
#
# For example, the name for the first multiscales entry isn't very descriptive,
# so lets update it

metadata.multiscales[0].name = "A cat"
pprint(metadata.multiscales)

# One constraint in the OME-zarr spec is that the coordinate transforms have to be a
# scale, or a scale then translation (strictly in that order). So if we try and make a
# transformation just a translation, it will raise an error.

try:
    metadata.multiscales[0].datasets[0].coordinateTransformations = VectorTranslation(
        type="translation", translation=[1, 2, 3]
    )
except ValidationError as e:
    pprint(e)


# This means validation happens early, allowing you to catch errors
# before getting too far.

# ## Saving modified metadata
#
# Once we've modified the metadata, we can save it back to the zarr group by
# assigning to the `.attributes` property.

ome_zarr_image.attributes = metadata
pprint(ome_zarr_image.group.attrs.asdict())

# ## Accessing data
#
# Although these models do not handle reading or writing data, they do expose the zarr
# arrays.

zarr_arr = ome_zarr_image.group[metadata.multiscales[0].datasets[0].path]
pprint(zarr_arr)

# ## Not using validation
#
# If you want to create models that are not validated against the OME-zarr
# specifciation, you can use the ``model_construct`` method on the models.
