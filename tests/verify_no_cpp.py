import sys
import os

# Ensure we are looking at the local src
sys.path.insert(0, os.path.abspath("src"))

try:
    from opengnc.kalman_filters import MEKF, UKF_Attitude, ACCELERATION_AVAILABLE

    print(f"Acceleration available: {ACCELERATION_AVAILABLE}")

    # Test MEKF instantiation
    mekf = MEKF()
    print("MEKF initialized successfully.")

    # Test UKF instantiation
    ukf = UKF_Attitude()
    print("UKF_Attitude initialized successfully.")

    print("SUCCESS: Core filters are usable in pure Python mode.")
except Exception as e:
    print(f"FAILURE: Could not initialize filters in Python mode. Error: {e}")
    sys.exit(1)
