"""Virtual Arduino emulator implementing LED and Buzzer payload with real serial protocol."""
from dataclasses import dataclass
from enum import Enum
import platform
import threading
import time
from typing import Any, Dict, Optional, Tuple


class LEDColor(Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED = "RED"
    OFF = "OFF"


class BuzzerMode(Enum):
    OFF = 0
    INTERMITTENT = 1 # Warning: pulsed 800Hz beep
    CONTINUOUS = 2   # Critical: continuous high-pitched 2000Hz alert tone


@dataclass
class ArduinoPacket:
    """Represents a structured serial telemetry packet exchanged with Arduino payload."""
    alert_level: str
    led_color: LEDColor
    buzzer_mode: BuzzerMode
    timestamp: float

    def to_serial_string(self) -> str:
        """Serializes to the hardware ASCII packet format."""
        return f"<ALERT:{self.alert_level},LED:{self.led_color.value},BUZ:{self.buzzer_mode.value}>\n"

    @classmethod
    def from_serial_string(cls, raw: str) -> Optional['ArduinoPacket']:
        """Parses an incoming hardware ASCII packet."""
        raw = raw.strip()
        if not (raw.startswith("<") and raw.endswith(">")):
            return None
        content = raw[1:-1]
        tokens = content.split(",")
        data = {}
        for token in tokens:
            if ":" in token:
                k, v = token.split(":", 1)
                data[k.strip()] = v.strip()

        alert = data.get("ALERT", "NORMAL")
        led = LEDColor(data.get("LED", "GREEN"))
        buz = BuzzerMode(int(data.get("BUZ", 0)))
        return cls(alert_level=alert, led_color=led, buzzer_mode=buz, timestamp=time.time())


class VirtualArduino:
    """Software emulation of the Arduino Nano / Uno payload board.

    Exposes the exact serial protocol used by the physical hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        al_cfg = config.get("alerts", {})
        self.use_hardware_serial = al_cfg.get("use_hardware_serial", False)
        self.serial_port = al_cfg.get("serial_port", "COM3")
        self.baud_rate = al_cfg.get("baud_rate", 115200)

        va_cfg = al_cfg.get("virtual_arduino", {})
        self.enable_audio = va_cfg.get("enable_audio", False) # Silent by default to avoid beep spam

        self.led_color = LEDColor.GREEN
        self.buzzer_mode = BuzzerMode.OFF
        self.active_alert = "NORMAL"

        self.last_tx_packet: Optional[str] = None
        self.rx_buffer: str = ""
        self.tx_history = []

        self._serial_conn = None
        if self.use_hardware_serial:
            self._init_hardware_serial()

    def _init_hardware_serial(self) -> None:
        """Attempt to connect to real hardware Arduino via PySerial."""
        try:
            import serial
            self._serial_conn = serial.Serial(self.serial_port, self.baud_rate, timeout=0.1)
            print(f"[VirtualArduino] Connected to physical Arduino on {self.serial_port}")
        except Exception as e:
            print(f"[VirtualArduino] Physical serial connection unavailable ({e}). Running in virtual mode.")
            self._serial_conn = None

    def update_alert(self, alert_level_str: str) -> ArduinoPacket:
        """Compute the corresponding LED color and Buzzer mode for the alert level,

        then transmit the serial packet.
        """
        self.active_alert = alert_level_str

        if alert_level_str == "CRITICAL":
            self.led_color = LEDColor.RED
            self.buzzer_mode = BuzzerMode.CONTINUOUS
        elif alert_level_str == "WARNING":
            self.led_color = LEDColor.AMBER
            self.buzzer_mode = BuzzerMode.INTERMITTENT
        else:
            self.led_color = LEDColor.GREEN
            self.buzzer_mode = BuzzerMode.OFF

        packet = ArduinoPacket(
            alert_level=alert_level_str,
            led_color=self.led_color,
            buzzer_mode=self.buzzer_mode,
            timestamp=time.time()
        )

        serial_str = packet.to_serial_string()
        self.last_tx_packet = serial_str
        self.tx_history.append(serial_str)
        if len(self.tx_history) > 50:
            self.tx_history.pop(0)

        # Send over physical serial if active
        if self._serial_conn is not None:
            try:
                self._serial_conn.write(serial_str.encode("ascii"))
            except Exception:
                pass

        return packet

    def get_ui_led_rgb(self) -> Tuple[int, int, int]:
        """Returns BGR color tuple for the OpenCV HUD display."""
        if self.led_color == LEDColor.RED:
            return (40, 40, 240) # Bright Red BGR
        elif self.led_color == LEDColor.AMBER:
            return (30, 160, 240) # Amber / Orange BGR
        elif self.led_color == LEDColor.GREEN:
            return (40, 220, 40) # Bright Green BGR
        return (100, 100, 100) # Off / Gray

    def get_buzzer_status_text(self) -> str:
        """Returns readable buzzer state for HUD log."""
        if self.buzzer_mode == BuzzerMode.CONTINUOUS:
            return "BUZZER: CONTINUOUS [2000Hz]"
        elif self.buzzer_mode == BuzzerMode.INTERMITTENT:
            return "BUZZER: PULSED [800Hz]"
        return "BUZZER: MUTED"
