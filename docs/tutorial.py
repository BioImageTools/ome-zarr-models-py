# # Tutorial

import zarr
import zarr.storage
from rich.pretty import pprint

from ome_zarr_models.v04 import Image
from ome_zarr_models.v04.coordinate_transformations import (
    VectorTranslation,
)

# ## Creating models
#
# We can create an Image model from a zarr group, that points to an
# OME-zarr dataset:

group = zarr.open("https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr")
ome_zarr_image = Image.from_zarr(group)
pprint(ome_zarr_image)

# This image contains both the zarr group, and a model of the multiscales metadata

multiscales_meta = ome_zarr_image.attributes.multiscales
pprint(multiscales_meta)

# ## Updating models
#
# All the fields in the models can be updated in place. When you do this, any
# validation on the individual field you are updating will take place.
#
# For example, there is no name for the first multiscales entry, so lets add it

multiscales_meta[0].name = "The first multiscales entry"
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
