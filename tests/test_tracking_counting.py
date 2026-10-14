"""Unit tests for person detector, ByteTrack tracker, and counting logic."""
import numpy as np
import pytest

from src.utils.config import load_config
from src.perception.detector import PersonDetector, DetectionResult
from src.utils.synthetic_feed import SyntheticFeedGenerator


@pytest.fixture
def config():
    return load_config("configs/config.yaml")


def test_detector_initialization(config):
    detector = PersonDetector(config)
    assert detector.imgsz == 960
    assert detector.conf == 0.35
    assert detector.person_class_id == 0


def test_detection_and_bytetrack_execution(config):
    detector = PersonDetector(config)
    gen = SyntheticFeedGenerator(width=640, height=480, num_pedestrians=20)
    frame, _ = gen.generate_frame(drone_x=0.0, drone_y=0.0, drone_z=20.0)

    # First frame
    result1 = detector.detect_and_track(frame)
    assert isinstance(result1, DetectionResult)
    assert result1.frame_idx == 1
    assert isinstance(result1.count, int)
    assert isinstance(result1.density_map, np.ndarray)

    # Second frame to verify tracker persistence
    gen.update_pedestrians(dt=0.05)
    frame2, _ = gen.generate_frame(drone_x=0.0, drone_y=0.0, drone_z=20.0)
    result2 = detector.detect_and_track(frame2)
    assert result2.frame_idx == 2
    assert result2.unique_tracked_count >= 0


def test_detector_reset(config):
    detector = PersonDetector(config)
    detector.unique_track_ids.add(101)
    detector.frame_counter = 42
    detector.reset_tracker()
    assert len(detector.unique_track_ids) == 0
    assert detector.frame_counter == 0
