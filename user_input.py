import inputs
import importlib
import threading
import time

# Try to import keyboard - may fail over SSH without X display
try:
    from pynput import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError as e:
    print(f"Keyboard support disabled: {e}")
    KEYBOARD_AVAILABLE = False

# ——————————————————————————————————————————————————————————————
# Map raw ev.code → friendly button name
_CODE_TO_NAME = {
    "BTN_SOUTH":     "A",
    "BTN_EAST":      "B",
    "BTN_NORTH":     "Y",
    "BTN_WEST":      "X",
    "BTN_TL":        "LB",
    "BTN_TR":        "RB",
    "BTN_TL2":       "LT",
    "BTN_TR2":       "RT",
    "BTN_SELECT":    "SLCT",
    "BTN_START":     "START",
    "BTN_MODE":      "GUIDE",
    "BTN_THUMBL":    "L3",
    "BTN_THUMBR":    "R3",
    "BTN_DPAD_UP":   "DPAD_UP",
    "BTN_DPAD_DOWN": "DPAD_DOWN",
    "BTN_DPAD_LEFT": "DPAD_LEFT",
    "BTN_DPAD_RIGHT":"DPAD_RIGHT",
}

# Map raw absolute codes → friendly axis names
_ABS_TO_NAME = {
    'ABS_X':     'LX',      # left stick X
    'ABS_Y':     'LY',      # left stick Y
    'ABS_RX':    'RX',      # right stick X
    'ABS_RY':    'RY',      # right stick Y
    'ABS_Z':     'LT',      # left trigger
    'ABS_RZ':    'RT',      # right trigger
    'ABS_HAT0X': 'DPAD_X',  # D-pad horizontal
    'ABS_HAT0Y': 'DPAD_Y',  # D-pad vertical
}

# All known friendly names — used by _parse_input to disambiguate e.g. "L3" vs "L" on device 3
_ALL_KNOWN_NAMES = set(_CODE_TO_NAME.values()) | set(_ABS_TO_NAME.values())

# Global configuration
INVERT_Y_AXIS = False  # Set to True to invert Y-axis controls (LY, RY)

# ——————————————————————————————————————————————————————————————
# Per-device gamepad state (keyed by int device index)
_pressed_buttons:       dict[int, set]  = {}   # idx -> set of held button names
_just_pressed_buttons:  dict[int, set]  = {}   # idx -> set of just-pressed names
_just_released_buttons: dict[int, set]  = {}   # idx -> set of just-released names
_toggles:               dict[int, dict] = {}   # idx -> {button_name: bool}
_axis_states:           dict[int, dict] = {}   # idx -> {axis_name: raw_value}

# Thread management
_pad_threads: dict[int, threading.Thread] = {}
_pad_stops:   dict[int, threading.Event]  = {}
_pad_lock = threading.Lock()

# ——————————————————————————————————————————————————————————————
# Global keyboard state
pressed_keys       = set()
just_pressed_keys  = set()
just_released_keys = set()
toggles            = {}

# ——————————————————————————————————————————————————————————————
# Internal helpers

def _repr_button(code: str) -> str:
    """Return a friendly name for a raw event code."""
    return _CODE_TO_NAME.get(code, code)

def _ensure_device_state(idx: int):
    """Create per-device state containers for index idx if not yet present."""
    if idx not in _pressed_buttons:
        _pressed_buttons[idx]       = set()
        _just_pressed_buttons[idx]  = set()
        _just_released_buttons[idx] = set()
        _toggles[idx]               = {}
        _axis_states[idx]           = {}

def _parse_input(input_name: str) -> tuple[str, int]:
    """Parse an input specifier into (name, device_index).

    'A'   -> ('A',  0)    plain name defaults to device 0
    'A1'  -> ('A',  1)    trailing digit selects device
    'L3'  -> ('L3', 0)    'L' is not a known name, no stripping
    'L31' -> ('L3', 1)    'L3' is known, so '1' is the device index
    """
    if input_name and input_name[-1].isdigit():
        candidate = input_name[:-1]
        if candidate in _ALL_KNOWN_NAMES:
            return candidate, int(input_name[-1])
    return input_name, 0

# ——————————————————————————————————————————————————————————————
# Event processing (shared by all reader threads)

def _process_events(events, idx: int):
    pb  = _pressed_buttons[idx]
    jpb = _just_pressed_buttons[idx]
    jrb = _just_released_buttons[idx]
    tgl = _toggles[idx]
    axs = _axis_states[idx]

    for ev in events:
        # DIGITAL BUTTONS
        if ev.ev_type == 'Key':
            name = _repr_button(ev.code)
            if ev.state == 1:
                if name not in pb:
                    jpb.add(name)
                    tgl[name] = not tgl.get(name, False)
                pb.add(name)
            else:
                if name in pb:
                    jrb.add(name)
                pb.discard(name)

        # ANALOG AXES & BINARY MAPPINGS
        elif ev.ev_type == 'Absolute':
            axis = _ABS_TO_NAME.get(ev.code)
            if not axis:
                continue
            value = ev.state
            axs[axis] = value

            # TRIGGERS as binary buttons
            if axis in ('LT', 'RT'):
                if value > 0:
                    if axis not in pb: jpb.add(axis)
                    pb.add(axis)
                else:
                    if axis in pb: jrb.add(axis)
                    pb.discard(axis)

            # D-PAD as binary buttons
            elif axis == 'DPAD_X':
                if value < 0:
                    if 'DPAD_LEFT' not in pb: jpb.add('DPAD_LEFT')
                    pb.add('DPAD_LEFT')
                else:
                    if 'DPAD_LEFT' in pb: jrb.add('DPAD_LEFT')
                    pb.discard('DPAD_LEFT')
                if value > 0:
                    if 'DPAD_RIGHT' not in pb: jpb.add('DPAD_RIGHT')
                    pb.add('DPAD_RIGHT')
                else:
                    if 'DPAD_RIGHT' in pb: jrb.add('DPAD_RIGHT')
                    pb.discard('DPAD_RIGHT')

            elif axis == 'DPAD_Y':
                if value < 0:
                    if 'DPAD_UP' not in pb: jpb.add('DPAD_UP')
                    pb.add('DPAD_UP')
                else:
                    if 'DPAD_UP' in pb: jrb.add('DPAD_UP')
                    pb.discard('DPAD_UP')
                if value > 0:
                    if 'DPAD_DOWN' not in pb: jpb.add('DPAD_DOWN')
                    pb.add('DPAD_DOWN')
                else:
                    if 'DPAD_DOWN' in pb: jrb.add('DPAD_DOWN')
                    pb.discard('DPAD_DOWN')

# ——————————————————————————————————————————————————————————————
# Per-device reader thread

def _reader_thread(device, idx: int, stop: threading.Event):
    while not stop.is_set():
        try:
            events = device.read()
        except Exception:
            break
        _process_events(events, idx)
    # Clear stale state on disconnect
    _pressed_buttons[idx].clear()
    _axis_states[idx].clear()

# ——————————————————————————————————————————————————————————————
# Watcher thread — detects connect/disconnect, manages reader threads

def _watcher_loop():
    _warned = False
    while True:
        try:
            pads = inputs.devices.gamepads
        except Exception:
            pads = []

        if not pads:
            if not _warned:
                print("Gamepad not connected. Waiting for connection...")
                _warned = True
            importlib.reload(inputs)
            time.sleep(1)
            continue

        if _warned:
            print(f"{len(pads)} gamepad(s) connected.")
            _warned = False

        with _pad_lock:
            current = set(range(len(pads)))
            # Start threads for new / restarted devices
            for i, pad in enumerate(pads):
                if i not in _pad_threads or not _pad_threads[i].is_alive():
                    _ensure_device_state(i)
                    stop = threading.Event()
                    t = threading.Thread(target=_reader_thread, args=(pad, i, stop), daemon=True)
                    _pad_stops[i] = stop
                    _pad_threads[i] = t
                    t.start()
            # Stop threads for devices that disappeared
            for i in set(_pad_threads.keys()) - current:
                _pad_stops[i].set()
                del _pad_threads[i]
                del _pad_stops[i]

        time.sleep(1)

# Start the watcher thread
_watcher = threading.Thread(target=_watcher_loop, daemon=True)
_watcher.start()

# Start the keyboard listener (only if available)
if KEYBOARD_AVAILABLE:
    def _on_press(key):
        key_repr = key if isinstance(key, str) else key.char if hasattr(key, 'char') else str(key)
        # Filter out None values before adding to sets
        if key_repr is None:
            return
        # Only mark as just pressed if it wasn't already noted as down.
        if key_repr not in pressed_keys:
            just_pressed_keys.add(key_repr)
        pressed_keys.add(key_repr)
        # Toggle state if applicable.
        if key_repr in toggles:
            toggles[key_repr] = not toggles[key_repr]

    def _on_release(key):
        key_repr = key if isinstance(key, str) else key.char if hasattr(key, 'char') else str(key)
        # Filter out None values
        if key_repr is None:
            return
        pressed_keys.discard(key_repr)
        # Mark the key as just released for falling edge detection.
        just_released_keys.add(key_repr)

    keyboard.Listener(on_press=_on_press, on_release=_on_release).start()

# ——————————————————————————————————————————————————————————————
# Public API (unified for both gamepad and keyboard)
#
# Gamepad input names accept an optional device-index suffix:
#   'A'      -> button A on device 0
#   'A1'     -> button A on device 1
#   'LX'     -> left-stick X on device 0
#   'LX1'    -> left-stick X on device 1
#   'L3'     -> L3 button on device 0  (no ambiguity: 'L' is not a known name)
#   'L31'    -> L3 button on device 1

def is_pressed(input_name):
    """True as long as the given button/key is held down."""
    name, idx = _parse_input(input_name)
    if idx in _pressed_buttons and name in _pressed_buttons[idx]:
        return True
    return name in pressed_keys

def is_toggled(input_name):
    """Flip-flop state for each press."""
    name, idx = _parse_input(input_name)
    if idx in _toggles and name in _toggles[idx]:
        return _toggles[idx][name]
    if name not in toggles:
        toggles[name] = False
    return toggles.get(name, False)

def rising_edge(input_name):
    """True exactly once when the button/key first goes down."""
    name, idx = _parse_input(input_name)
    if idx in _just_pressed_buttons and name in _just_pressed_buttons[idx]:
        _just_pressed_buttons[idx].discard(name)
        return True
    if name in just_pressed_keys:
        just_pressed_keys.discard(name)
        return True
    return False

def falling_edge(input_name):
    """True exactly once when the button/key first goes up."""
    name, idx = _parse_input(input_name)
    if idx in _just_released_buttons and name in _just_released_buttons[idx]:
        _just_released_buttons[idx].discard(name)
        return True
    if name in just_released_keys:
        just_released_keys.discard(name)
        return True
    return False

def get_axis(axis_name: str, normalize: bool = True) -> float:
    """Return axis state. Supports device-index suffix: 'LX', 'LX1', etc."""
    name, idx = _parse_input(axis_name)
    val = _axis_states.get(idx, {}).get(name, 0)
    if not normalize:
        return val
    # sticks: -32768..32767 -> -1.0..1.0
    if name in ("LX", "LY", "RX", "RY"):
        normalized = val / (32767.0 if val >= 0 else 32768.0)
        if INVERT_Y_AXIS and name in ("LY", "RY"):
            normalized = -normalized
        return round(normalized, 1)
    # triggers: 0..255 or 0..1023 -> 0.0..1.0
    if name in ("LT", "RT"):
        maxv = 255 if val <= 255 else 1023
        return val / maxv
    # D-pad: already -1, 0, 1
    if name == "DPAD_Y":
        return -val
    if name == "DPAD_X":
        return val
    return val

# Helpers for common use cases
def get_bipolar_ctrl(high_key=None, low_key=None, high_game=None, low_game=None, keyboard_scale=1.0, game_scale=1.0):
    """Returns -1.0 to +1.0. Combines gamepad and keyboard inputs with clamping.
    high_game/low_game accept axis or button names with optional device-index suffix.
    keyboard_scale: multiplier for keyboard inputs (default 1.0).
    game_scale: multiplier for gamepad inputs (default 1.0)."""
    def _is_axis(v):
        if not v:
            return False
        n, _ = _parse_input(v)
        return n in _ABS_TO_NAME.values()

    high_val = get_axis(high_game or '') if _is_axis(high_game) else int(is_pressed(high_game or ''))
    low_val  = get_axis(low_game  or '') if _is_axis(low_game)  else int(is_pressed(low_game  or ''))
    game_in  = (high_val - low_val) * game_scale
    key_in   = (int(is_pressed(high_key or '')) - int(is_pressed(low_key or ''))) * keyboard_scale
    return float(max(-1, min(1, game_in + key_in)))

# if __name__ == '__main__':
#     while True:
#         print(get_bipolar_ctrl('w', 's', 'RY'))
#         time.sleep(0.1)

# Example usage
if __name__ == '__main__':
    print("Input Manager Test - Press keys/buttons or move axes...")
    print("Press Ctrl+C to exit\n")

    while True:
        active_inputs = []

        # Show pressed buttons per gamepad
        for idx in sorted(_pressed_buttons.keys()):
            if _pressed_buttons[idx]:
                active_inputs.append(f"[pad {idx}] Buttons: {', '.join(sorted(_pressed_buttons[idx]))}")

        # Show pressed keyboard keys
        if pressed_keys:
            active_inputs.append(f"Keys: {', '.join(sorted(pressed_keys))}")

        # Show non-zero axes per gamepad
        for idx in sorted(_axis_states.keys()):
            active_axes = []
            for axis_name in ["LX", "LY", "RX", "RY", "LT", "RT", "DPAD_X", "DPAD_Y"]:
                value = get_axis(f"{axis_name}{idx}", normalize=True)
                if abs(value) > 0.01:
                    active_axes.append(f"{axis_name}: {value:+.2f}")
            if active_axes:
                active_inputs.append(f"[pad {idx}] Axes: {', '.join(active_axes)}")

        if active_inputs:
            print(" | ".join(active_inputs))

        time.sleep(0.1)
