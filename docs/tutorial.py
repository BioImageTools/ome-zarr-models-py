# # Tutorial

import zarr
import zarr.storage
from rich.pretty import pprint

from ome_zarr_models.data import tutorial_data_path
from ome_zarr_models.v04 import Image
from ome_zarr_models.v04.coordinate_transformations import (
    VectorTranslation,
)

# ## Creating models
#
# We can create an Image model from a zarr group, that points to an
# OME-zarr dataset:

print(tutorial_data_path.absolute())
group = zarr.open(tutorial_data_path)
ome_zarr_image = Image(group)
pprint(ome_zarr_image)

# This image contains both the zarr group, and a model of the multiscales metadata

multiscales_meta = ome_zarr_image.attributes.multiscales
pprint(multiscales_meta)

# ## Updating models
#
# All the fields in the models can be updated in place. When you do this, any
# validation on the individual field you are updating will take place.
#
# For example, the name for the first multiscales entry isn't very descriptive,
# so lets update it

multiscales_meta[0].name = "A cat"
pprint(multiscales_meta)

# One constraint in the OME-zarr spec is that the coordinate transforms have to be a
# scale, or a scale then translation (strictly in that order). So if we try and make a
# transformation just a translation, it will raise an error.

multiscales_meta[0].datasets[0].coordinateTransformations = VectorTranslation(
    type="translation", translation=[1, 2, 3]
)


# This means validation happens early, allowing you to catch errors
# before getting too far.

# ## Accessing data
#
# Although these models do not handle reading or writing data, they do expose the zarr
# arrays.

zarr_arr = ome_zarr_image.group[multiscales_meta[0].datasets[0].path]
pprint(zarr_arr)

# ## Not using validation
#
# If you want to create models that are not validated against the OME-zarr
# specifciation, you can use the ``model_construct`` method on the models.
