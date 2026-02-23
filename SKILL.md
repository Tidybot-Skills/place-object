---
name: tb-place-object
description: Visual servoing place skill — places a held object onto a detected target surface/object. Use when (1) the robot is holding an object and needs to place it somewhere, (2) chaining as a place primitive after tb-pick-up-object, (3) the user says "put it on the plate" or similar.
---

# Place Object

Raises held object to home height (camera sees past it), detects place target via mask-only detection, servos laterally to center above target, descends while tracking, releases.

## Pipeline

1. Go home (holding object high)
2. Tilt camera down
3. Detect place target (mask-only — ignores bbox-only detections)
4. Servo laterally to center above target
5. Descend to place height while tracking
6. Release, go home

## Usage

```python
from main import place_object
place_object(target="red plate")
```

Assumes robot is already holding an object (e.g., after `tb-pick-up-object`).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| PLACE_TARGET | "red plate" | Default place target |
| CAMERA_ID | "309622300814" | Wrist camera ID |
| PLACE_Z | -0.35 | Target place height (meters) |
| PLACE_PIXEL_TOLERANCE | 40 | Centering tolerance (pixels) |
| CAMERA_TILT_RAD | -20° | Camera tilt for detection |

## Dependencies

None (uses robot_sdk directly). See `scripts/deps.txt`.
