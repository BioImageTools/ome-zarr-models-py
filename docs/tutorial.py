# # Tutorial

import matplotlib.pyplot as plt
import zarr
import zarr.storage
from rich.pretty import pprint

from ome_zarr_models.v04 import Image

# ## Loading datasets
#
# OME-zarr datasets are just zarr groups with special metadata.
# To open an OME-zarr dataset, we first open the zarr group, and
# then create an image object from it. This will validate the
# metadata.

group = zarr.open(
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr", mode="r"
)
ome_zarr_image = Image.from_zarr(group)
pprint(ome_zarr_image)

# No errors, which means the metadata is valid 🎉
#
# ## Accessing metadata
# To access the OME-zarr metadata, use the `.attributes` property:

metadata = ome_zarr_image.attributes
pprint(metadata)

# And as an example of getting more specific metadata, lets get the metadata
# for all the datasets in this multiscales:

pprint(metadata.multiscales[0].datasets)

# ## Accessing data
#
# Although these models do not handle reading or writing data, they do expose the zarr
# arrays. For example, to get the highest resolution image:

zarr_arr = group[metadata.multiscales[0].datasets[0].path]
pprint(zarr_arr)

# To finish off, lets plot the first z-slice of the first channel of this data:
plt.imshow(zarr_arr[0, 0, :, :], cmap="gray")
