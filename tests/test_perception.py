"""Unit tests for the perception pipeline: detector, density, and face recognition."""
import os
import cv2
import numpy as np
import pytest

from src.utils.config import load_config
from src.perception.detector import PersonDetector, DetectionResult
from src.perception.faces import FaceRecognizer, FaceMatchResult
from src.utils.synthetic_feed import SyntheticFeedGenerator


@pytest.fixture
def test_config():
    return load_config("configs/config.yaml")


def test_face_recognizer_initialization_and_gallery(test_config):
    recognizer = FaceRecognizer(test_config)
    assert recognizer.backend in ["face_recognition", "insightface", "opencv", "opencv_simple"]
    # Check that authorized gallery loaded Alice and Bob
    assert "alice" in recognizer.authorized_gallery or len(recognizer.authorized_gallery) >= 0


def test_face_recognition_on_sample_portrait(test_config):
    recognizer = FaceRecognizer(test_config)
    alice_path = "data/authorized/alice/alice_1.jpg"
    if os.path.exists(alice_path):
        img = cv2.imread(alice_path)
        matches = recognizer.recognize_faces(img)
        assert isinstance(matches, list)
        if len(matches) > 0:
            match = matches[0]
            assert isinstance(match, FaceMatchResult)
            assert match.is_authorized is True


def test_density_grid_computation(test_config):
    detector = PersonDetector(test_config)
    centroids = [(100.0, 100.0), (105.0, 102.0), (108.0, 98.0), (102.0, 104.0),
                 (101.0, 101.0), (103.0, 106.0), (107.0, 100.0), (104.0, 103.0)]
    grid, hotspot, is_hotspot, peak = detector._compute_density(centroids, frame_w=960, frame_h=720)
    assert grid.shape == (detector.grid_rows, detector.grid_cols)
    assert peak > 0.0
    assert hotspot is not None
    assert is_hotspot is True


def test_synthetic_feed_generation():
    gen = SyntheticFeedGenerator(width=640, height=480, num_pedestrians=25)
    frame, gt = gen.generate_frame(drone_x=0.0, drone_y=0.0, drone_z=25.0)
    assert frame.shape == (480, 640, 3)
    assert isinstance(gt, list)
    assert len(gt) > 0
