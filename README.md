# place-object

Author: evilsky
Dependencies: (none)

Visual servoing place skill. Raises held object to home height, tilts camera down (sees past held object from height), detects place target using mask-only detection, servos laterally to center above target, descends while tracking, and releases.

Key features:
- **Raise and look around**: From home height, wrist camera sees past the held object — solves the occlusion problem
- **Mask-only detection**: Only trusts segmentation masks (ignores bbox-only detections) for robust tracking during descent
- **Live visual servoing**: Continuously tracks place target during both lateral centering and descent
- **Graceful degradation**: Places at current position if target mask is lost during descent

## Usage

```python
from main import place_object
place_object(target="red plate")
```

Assumes the robot is already holding an object (e.g. after `pick-up-object`).

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| PLACE_TARGET | "red plate" | Default place target |
| CAMERA_ID | "309622300814" | Wrist camera ID |
| PLACE_Z | -0.35 | Target place height (meters) |
| PLACE_PIXEL_TOLERANCE | 40 | Centering tolerance (pixels) |
| CAMERA_TILT_RAD | -20° | Camera tilt for detection |

## Pipeline

1. Go home (holding object high)
2. Tilt camera down
3. Detect place target (mask-only)
4. Servo laterally to center above target
5. Descend to place height while tracking
6. Release, go home
