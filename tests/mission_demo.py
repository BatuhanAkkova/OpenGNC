import time
import json
import numpy as np
import requests
import multiprocessing
from opengnc.ground_segment.ccsds import SpacePacket, CUC, PacketType
from opengnc.ground_segment.decom import DecomEngine, TelemetryField
from opengnc.ssa.collision_avoidance import plan_collision_avoidance_maneuver
from opengnc.dashboard.server import run_server


def run_demo():
    # 1. Define Telemetry Layout
    # Format: [Alt(f32), Vel(f32), q0(f32), q1(f32), q2(f32), q3(f32)]
    fields = [
        TelemetryField("alt", "f", 0),
        TelemetryField("vel", "f", 4),
        TelemetryField("q0", "f", 8),
        TelemetryField("q1", "f", 12),
        TelemetryField("q2", "f", 16),
        TelemetryField("q3", "f", 20),
    ]
    decom = DecomEngine(fields)

    print("[DEMO] Starting Mission Control Server in background...")
    # Start server in a separate process
    server_process = multiprocessing.Process(target=run_server, args=(8000,), daemon=True)
    server_process.start()

    time.sleep(2)  # Wait for server to start

    print("[DEMO] Simulating Satellite Telemetry...")

    # Simulation parameters
    t = 0
    dt = 1.0
    r_mag = 7000e3  # 7000 km
    v_mag = np.sqrt(3.986e14 / r_mag)

    try:
        while t < 60:  # Run for 60 seconds
            # Update state (circular orbit approx)
            angle = (v_mag / r_mag) * t
            lat = np.rad2deg(np.sin(angle) * 0.5)
            lon = np.rad2deg(angle) % 360 - 180

            alt = (r_mag - 6371e3) / 1000.0  # km
            vel = v_mag / 1000.0  # km/s
            q = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]

            # Create Binary Payload
            import struct

            payload_data = struct.pack(">ffffff", alt, vel, q[0], q[1], q[2], q[3])

            # Wrap in CCSDS with CUC secondary header
            # Total data = CUC (6 bytes) + payload_data
            cuc_bytes = CUC.pack()
            full_data = cuc_bytes + payload_data

            packet = SpacePacket(
                apid=0x123, packet_type=PacketType.TELEMETRY, sec_header_flag=True, data=full_data
            )
            raw_packet = packet.pack()

            # --- "DOWNLINK" ---
            # Ground station receives raw_packet
            rx_packet = SpacePacket.unpack(raw_packet)
            rx_cuc = CUC.unpack(rx_packet.data[:6])
            rx_payload = rx_packet.data[6:]

            tlm = decom.decommutate(rx_payload)
            tlm["lat"] = lat
            tlm["lon"] = lon
            tlm["pc"] = 1.2e-5 * (1 + 0.1 * np.sin(t / 10))  # Simulated Pc
            tlm["timestamp"] = rx_cuc

            # Push to Dashboard
            try:
                requests.post("http://localhost:8000/telemetry", json=tlm, timeout=0.1)
            except Exception:
                pass

            print(f"T={t:0.1f}s | Alt={alt:0.2f}km | Lat={lat:0.2f} | Lon={lon:0.2f}")

            time.sleep(dt)
            t += dt

    except KeyboardInterrupt:
        pass
    finally:
        print("[DEMO] Shutting down...")
        server_process.terminate()


if __name__ == "__main__":
    run_demo()
