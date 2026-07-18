"""
Demo of standardized interfaces for external tool interoperability.
"""

import numpy as np

from opengnc.interfaces.ccsds import OEM, OPM


def main() -> None:
    """Create sample CCSDS interchange artifacts and read them back."""
    print("--- OpenGNC Interoperability Demo ---")

    state_kms = {
        "EPOCH": "2026-04-18T12:00:00.000",
        "X": 6678.14,
        "Y": 0.0,
        "Z": 0.0,
        "X_DOT": 0.0,
        "Y_DOT": 7.5,
        "Z_DOT": 0.0,
    }

    opm = OPM(state_kms)
    opm_path = "mission_initial_conditions.opm"
    opm.write(opm_path)
    print(f"Created OPM file: {opm_path}")

    epochs = [
        "2026-04-18T12:00:00.000",
        "2026-04-18T12:10:00.000",
        "2026-04-18T12:20:00.000",
    ]
    states = np.array(
        [
            [6678.14, 0.0, 0.0, 0.0, 7.5, 0.0],
            [6500.0, 1500.0, 0.0, -1.0, 7.3, 0.0],
            [6000.0, 3000.0, 0.0, -2.0, 6.8, 0.0],
        ]
    )

    oem = OEM()
    oem.set_data(epochs, states)
    oem_path = "mission_trajectory.oem"
    oem.write(oem_path)
    print(f"Created OEM file: {oem_path}")

    oem_loaded = OEM.from_file(oem_path)
    print(f"Loaded OEM with {len(oem_loaded.data)} points.")
    print(f"Reference Frame: {oem_loaded.metadata['REF_FRAME']}")


if __name__ == "__main__":
    main()
