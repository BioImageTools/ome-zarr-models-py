from collections.abc import Generator
from typing import Self

from pydantic import model_validator
from pydantic_zarr.v2 import ArraySpec, GroupSpec

from ome_zarr_models.base import Base
from ome_zarr_models.v04.plate import Plate
from ome_zarr_models.v04.well import WellGroup

__all__ = ["HCSAttrs"]


class HCSAttrs(Base):
    """
    HCS metadtata attributes.
    """

    plate: Plate


class HCS(GroupSpec[HCSAttrs, ArraySpec | GroupSpec]):
    """
    An OME-zarr high-content screening (HCS) dataset representing a single plate.
    """

    @model_validator(mode="after")
    def _check_valid_acquisitions(self) -> Self:
        """
        Check well acquisition IDs are in list of plate acquisition ids.
        """
        acquisitions = self.attributes.plate.acquisitions
        if acquisitions is None:
            return self

        valid_aq_ids = [aq.id for aq in acquisitions]

        for well_i, well_group in enumerate(self.well_groups):
            for image_i, well_image in enumerate(well_group.attributes.well.images):
                if well_image.acquisition is None:
                    continue
                elif well_image.acquisition not in valid_aq_ids:
                    msg = (
                        f"Acquisition ID '{well_image.acquisition} "
                        f"(found in well {well_i}, {image_i}) "
                        f"is not in list of plate acquisitions: {valid_aq_ids}"
                    )
                    raise ValueError(msg)

        return self

    def get_well_group(self, i: int) -> WellGroup:
        """
        Get a single well group.
        """
        well = self.attributes.plate.wells[i]
        well_path = well.path
        well_path_parts = well_path.split("/")
        group = self
        for part in well_path_parts:
            group = group.members[part]

        return WellGroup(attributes=group.attributes, members=group.members)

    @property
    def n_wells(self) -> int:
        """
        Number of wells.
        """
        return len(self.attributes.plate.wells)

    @property
    def well_groups(self) -> Generator[WellGroup, None, None]:
        for i in range(self.n_wells):
            yield self.get_well_group(i)
