"""Simulation backends supporting Webots, Video files, Webcams, and MAVLink."""
from .base import SimulatorBackend
from .video_backend import VideoFileBackend
from .webcam_backend import WebcamBackend
from .webots_backend import WebotsBackend
from .mavlink_stub import MAVLinkBackendStub

__all__ = [
    "SimulatorBackend",
    "VideoFileBackend",
    "WebcamBackend",
    "WebotsBackend",
    "MAVLinkBackendStub",
]
