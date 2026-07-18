"""
Utility for managing and searching star catalogs.
"""

from typing import TypedDict

import numpy as np


class _StarEntry(TypedDict):
    id: int
    mag: float
    vec: np.ndarray


class StarCatalog:
    """
    Utility for managing and searching star catalogs (e.g., Hipparcos).
    """

    def __init__(self, catalog_path: str | None = None) -> None:
        """
        Args:
            catalog_path (str): Path to the catalog file (CSV/DAT).
        """
        self.stars: list[_StarEntry] = []
        if catalog_path:
            self.load_catalog(catalog_path)

    def load_catalog(self, path: str) -> None:
        """
        Dummy implementation for catalog loading.
        In a real scenario, this would parse Hipparcos or similar data.
        """
        # Create a few synthetic stars for testing
        self.stars = [
            {"id": 1, "mag": 0.0, "vec": np.array([1.0, 0.0, 0.0])},  # Vega-ish
            {"id": 2, "mag": 0.5, "vec": np.array([0.0, 1.0, 0.0])},
            {"id": 3, "mag": 1.0, "vec": np.array([0.0, 0.0, 1.0])},
            {"id": 4, "mag": 0.1, "vec": np.array([-1.0, 0.0, 0.0])},
        ]

    def get_stars_in_fov(
        self, boresight: np.ndarray, fov_deg: float, min_mag: float | None = None
    ) -> list[_StarEntry]:
        """
        Filters stars within a given Field of View (FOV).

        Args:
            boresight (np.ndarray): Unit vector of the camera boresight in J2000.
            fov_deg (float): Full Field of View in degrees.
            min_mag (float): Minimum magnitude (brightness threshold).

        Returns
        -------
            list: Stars within the FOV.
        """
        cos_limit = np.cos(np.radians(fov_deg / 2))
        visible_stars: list[_StarEntry] = []

        for star in self.stars:
            if min_mag is not None and star["mag"] > min_mag:
                continue

            cos_alpha = np.dot(boresight, star["vec"])
            if cos_alpha >= cos_limit:
                visible_stars.append(star)

        return visible_stars




