"""Unit tests for Virtual Arduino emulator and serial packet communication."""
import pytest
from src.utils.config import load_config
from src.alerts.virtual_arduino import VirtualArduino, ArduinoPacket, LEDColor, BuzzerMode


@pytest.fixture
def arduino():
    cfg = load_config("configs/config.yaml")
    return VirtualArduino(cfg)


def test_packet_serialization_and_parsing():
    pkt = ArduinoPacket(alert_level="CRITICAL", led_color=LEDColor.RED, buzzer_mode=BuzzerMode.CONTINUOUS, timestamp=100.0)
    serial_str = pkt.to_serial_string()
    assert serial_str == "<ALERT:CRITICAL,LED:RED,BUZ:2>\n"

    parsed = ArduinoPacket.from_serial_string(serial_str)
    assert parsed is not None
    assert parsed.alert_level == "CRITICAL"
    assert parsed.led_color == LEDColor.RED
    assert parsed.buzzer_mode == BuzzerMode.CONTINUOUS


def test_alert_state_mappings(arduino):
    # Normal
    p_norm = arduino.update_alert("NORMAL")
    assert arduino.led_color == LEDColor.GREEN
    assert arduino.buzzer_mode == BuzzerMode.OFF
    assert p_norm.to_serial_string() == "<ALERT:NORMAL,LED:GREEN,BUZ:0>\n"

    # Warning
    p_warn = arduino.update_alert("WARNING")
    assert arduino.led_color == LEDColor.AMBER
    assert arduino.buzzer_mode == BuzzerMode.INTERMITTENT
    assert p_warn.to_serial_string() == "<ALERT:WARNING,LED:AMBER,BUZ:1>\n"

    # Critical
    p_crit = arduino.update_alert("CRITICAL")
    assert arduino.led_color == LEDColor.RED
    assert arduino.buzzer_mode == BuzzerMode.CONTINUOUS
    assert p_crit.to_serial_string() == "<ALERT:CRITICAL,LED:RED,BUZ:2>\n"


def test_ui_color_rgb(arduino):
    arduino.update_alert("NORMAL")
    rgb = arduino.get_ui_led_rgb()
    assert isinstance(rgb, tuple) and len(rgb) == 3
    # Green dominant in BGR
    assert rgb[1] > rgb[0] and rgb[1] > rgb[2]
