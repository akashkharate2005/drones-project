"""Generate sample portraits, billboard textures, and a demonstration video clip."""
import sys
import os

# Ensure src is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.synthetic_feed import generate_sample_assets, generate_sample_video


def main():
    print("[1/2] Generating authorized & unknown portraits and billboards...")
    generate_sample_assets("data")
    print("      -> Saved data/authorized/alice/alice_1.jpg")
    print("      -> Saved data/authorized/bob/bob_1.jpg")
    print("      -> Saved data/unknown/suspect_charlie/suspect_1.jpg")
    print("      -> Saved data/billboards/ textures")

    print("[2/2] Rendering sample aerial surveillance video (data/samples/crowd_surveillance_demo.mp4)...")
    generate_sample_video("data/samples/crowd_surveillance_demo.mp4", num_frames=120)
    print("      -> Rendered demo surveillance video successfully.")
    print("All synthetic assets generated successfully!")


if __name__ == "__main__":
    main()
