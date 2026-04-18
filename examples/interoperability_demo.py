"""
Demo of standardized interfaces for external tool interoperability.
"""

import os

import numpy as np

from opengnc.interfaces.ccsds import OEM, OPM


def main():
    print("--- OpenGNC Interoperability Demo ---")

    # 1. Define an initial state for a satellite
    # E.g., Near-circular LEO
    state_kms = {
        "EPOCH": "2026-04-18T12:00:00.000",
        "X": 6678.14,
        "Y": 0.0,
        "Z": 0.0,
        "X_DOT": 0.0,
        "Y_DOT": 7.5,
        "Z_DOT": 0.0,
    }

    # 2. Save to CCSDS OPM (Orbit Parameter Message)
    # This file can be imported directly into GMAT or Orekit
    opm = OPM(state_kms)
    opm_path = "mission_initial_conditions.opm"
    opm.write(opm_path)
    print(f"Created OPM file: {opm_path}")

    # 3. Create a mock trajectory for OEM (Orbit Ephemeris Message)
    epochs = [
        "2026-04-18T12:00:00.000",
        "2026-04-18T12:10:00.000",
        "2026-04-18T12:20:00.000",
    ]
    # Simple propagation (mock)
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

    # 4. Read back the OEM and verify
    oem_loaded = OEM.from_file(oem_path)
    print(f"Loaded OEM with {len(oem_loaded.data)} points.")
    print(f"Reference Frame: {oem_loaded.metadata['REF_FRAME']}")

    # 5. Clean up demo files
    # os.remove(opm_path)
    # os.remove(oem_path)


if __name__ == "__main__":
    main()
 Broadway
