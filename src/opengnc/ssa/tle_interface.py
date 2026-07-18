"""
TLE Catalog Interface and Utilities.
"""

from ..propagators.sgp4_propagator import Sgp4Propagator


class TLEEntity:
    """
    Representation of a Two-Line Element (TLE) set for a satellite.

    Parameters
    ----------
    name : str
        Common name of the satellite.
    line1 : str
        First line of the TLE.
    line2 : str
        Second line of the TLE.
    """

    def __init__(self, name: str, line1: str, line2: str) -> None:
        """Initialize TLE entity and parse metadata."""
        self.name = name.strip()
        self.line1 = line1.rstrip()
        self.line2 = line2.rstrip()
        self.norad_id = self._parse_norad_id()

    def _parse_norad_id(self) -> str:
        """Extract NORAD catalog ID from Line 1."""
        if len(self.line1) > 7:
            return self.line1[2:7].strip()
        return "00000"

    def get_propagator(self) -> Sgp4Propagator:
        """
        Create a propagator instance for this TLE.

        Returns
        -------
        Sgp4Propagator
            SGP4 propagator initialized with TLE lines.
        """
        return Sgp4Propagator(self.line1, self.line2)


class TLECatalog:
    """
    Catalog for managing and searching a collection of satellite TLEs.
    """

    def __init__(self) -> None:
        """Initialize empty catalog."""
        self.satellites: list[TLEEntity] = []
        self._by_id: dict[str, TLEEntity] = {}
        self._by_name: dict[str, TLEEntity] = {}

    def add_tle(self, name: str, line1: str, line2: str) -> None:
        """
        Register a new TLE in the catalog.

        Parameters
        ----------
        name : str
            Satellite name.
        line1, line2 : str
            TLE data.
        """
        entity = TLEEntity(name, line1, line2)
        self.satellites.append(entity)
        self._by_id[entity.norad_id] = entity
        self._by_name[entity.name.upper()] = entity

    def load_from_txt(self, filepath: str) -> None:
        """
        Load TLE data from a standard text file.

        Supports both 2-line and 3-line (with name) TLE formats.

        Parameters
        ----------
        filepath : str
            Path to the TLE file.
        """
        with open(filepath) as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        i = 0
        while i < len(lines) - 1:
            l1, l2 = lines[i], lines[i+1]
            if l1.startswith("1") and l2.startswith("2"):
                # 3-line format (Name above 1/2) or 2rd/3rd line of 3-line
                if i > 0 and not lines[i-1].startswith(("1", "2")):
                    name = lines[i-1]
                else:
                    name = f"SAT_{l1[2:7]}"
                self.add_tle(name, l1, l2)
                i += 2
            else:
                i += 1

    def get_by_norad_id(self, norad_id: str | int) -> TLEEntity | None:
        """Lookup satellite by catalog ID."""
        return self._by_id.get(str(norad_id).strip())

    def get_by_name(self, name: str) -> TLEEntity | None:
        """Lookup satellite by name (case-insensitive)."""
        return self._by_name.get(name.strip().upper())

    def list_satellites(self) -> list[str]:
        """List names of all registered satellites."""
        return [s.name for s in self.satellites]




