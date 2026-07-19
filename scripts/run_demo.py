from pathlib import Path

from autoguard_drift.demo import run

if __name__ == "__main__":
    print(run(Path("outputs/fleet_drift.json")))
