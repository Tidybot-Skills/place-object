"""Place object skill.

Visual servoing place: raise to home, tilt camera, detect place target from
height (camera sees past held object), servo laterally to center above target,
descend while tracking with mask-only detection, release.

Assumes the robot is already holding an object (e.g. after pick-up-object).

Workflow:
  1. Go home (holding object high)
  2. Tilt camera down — from home height, camera sees past held object
  3. Detect place target using mask-only detection
  4. Servo laterally above place target (live detection)
  5. Descend to place height while tracking
  6. Release, go home

Usage:
  from main import place_object
  place_object(target="red plate")
"""

from robot_sdk import arm, gripper, sensors, yolo, display
from robot_sdk.arm import ArmError
import numpy as np
import time
import math

# ============================================================================
# Configuration
# ============================================================================

PLACE_TARGET = "red circular plate"

CAMERA_ID = "309622300814"
DETECTION_CONFIDENCE = 0.15

# --- Visual servoing gains ---
GAIN_U_TO_DY = -0.0006
GAIN_U_TO_DX = 0.0
GAIN_V_TO_DX = -0.0006
GAIN_V_TO_DZ = 0.0

# --- Servoing loop parameters ---
PIXEL_TOLERANCE = 30
MAX_SERVO_ITERATIONS = 200
MAX_LATERAL_STEP_M = 0.05
MIN_LATERAL_STEP_M = 0.001
SERVO_MOVE_DURATION = 0.5

# --- Descent parameters ---
DESCEND_PAUSE_PIXELS = 80

# --- Place parameters ---
PLACE_Z = -0.35
PLACE_DEPTH_THRESHOLD = 0.35  # Release when depth to target < this (meters)
PLACE_DESCEND_STEP_M = 0.03
PLACE_PIXEL_TOLERANCE = 40
PLACE_LOST_RETRIES = 5

# --- Camera tilt ---
CAMERA_TILT_RAD = math.radians(-20)


# ============================================================================
# Helper functions
# ============================================================================

def detect_object_mask_only(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target object, only return detections that have a valid mask.

    Ignores bbox-only detections. As long as the mask is visible, we have
    the target — robust to partial occlusion from held object.
    """
    result = yolo.segment_camera(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    # Filter to mask-only detections
    masked = [d for d in detections if d.mask is not None and (d.mask > 0.5).sum() > 0]
    if not masked:
        return None, None
    best = max(masked, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def detect_object_3d(target: str, confidence: float = DETECTION_CONFIDENCE):
    """Detect target with 3D depth. Returns best masked detection + depth."""
    result = yolo.segment_camera_3d(
        target, camera_id=CAMERA_ID, confidence=confidence,
        save_visualization=True, mask_format="npz",
    )
    detections = result.get_by_class(target)
    if not detections:
        detections = result.detections
    if not detections:
        return None, None
    masked = [d for d in detections if d.mask is not None and (d.mask > 0.5).sum() > 0]
    if not masked:
        return None, None
    best = max(masked, key=lambda d: d.area if d.area > 0 else
               (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
    return best, result.image_shape


def get_object_pixel_center(detection):
    """Get object center in pixel coordinates (mask centroid or bbox center)."""
    if detection.mask is not None:
        mask = detection.mask
        binary = (mask > 0.5).astype(np.float32)
        total = binary.sum()
        if total > 0:
            ys, xs = np.where(binary > 0)
            return float(xs.mean()), float(ys.mean())
    x1, y1, x2, y2 = detection.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def get_image_center(image_shape):
    """Return image center pixel (no gripper offset for place)."""
    h, w = image_shape[0], image_shape[1]
    return w / 2.0, h / 2.0


def pixel_error_to_ee_delta(u_err, v_err):
    """Convert pixel error to EE delta in base frame."""
    dx = GAIN_U_TO_DX * u_err + GAIN_V_TO_DX * v_err
    dy = GAIN_U_TO_DY * u_err
    dz = GAIN_V_TO_DZ * v_err
    step_norm = np.sqrt(dx**2 + dy**2 + dz**2)
    if step_norm > MAX_LATERAL_STEP_M:
        scale = MAX_LATERAL_STEP_M / step_norm
        dx *= scale
        dy *= scale
        dz *= scale
    return dx, dy, dz


# ============================================================================
# Servo above target then descend
# ============================================================================

def servo_above_place(target: str):
    """Servo laterally to center above the place target.

    Uses image center as target (no gripper offset) since we're high up.
    Only corrects lateral error, no descent.
    Mask-only detection: only tracks if mask is visible (ignores bbox-only).
    """
    print(f"\n--- Servo-Above: centering over '{target}' (mask-only) ---")
    display.show_text(f"Centering over {target}...")
    display.show_face("thinking")

    consecutive_misses = 0

    for i in range(MAX_SERVO_ITERATIONS):
        det, shape = detect_object_mask_only(target)

        if det is None:
            consecutive_misses += 1
            print(f"  Iter {i+1}: mask not detected "
                  f"({consecutive_misses}/{PLACE_LOST_RETRIES})")
            if consecutive_misses >= PLACE_LOST_RETRIES:
                print("  ERROR: Lost place target mask. Aborting place.")
                return False
            time.sleep(0.5)
            continue
        consecutive_misses = 0

        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_image_center(shape)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        ee_z = sensors.get_ee_position()[2]
        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[mask] Z={ee_z:.3f}m")

        if error_mag < PLACE_PIXEL_TOLERANCE:
            print(f"  Centered above target!")
            return True

        dx, dy, _ = pixel_error_to_ee_delta(u_err, v_err)

        step = np.sqrt(dx**2 + dy**2)
        if step < MIN_LATERAL_STEP_M:
            print(f"    Centered (step too small). Done!")
            return True

        print(f"    Move: dx={dx*1000:.1f}mm, dy={dy*1000:.1f}mm")
        arm.move_delta(dx=dx, dy=dy, dz=0, droll=0, dpitch=0, dyaw=0,
                       frame="base", duration=SERVO_MOVE_DURATION)
        time.sleep(0.2)

    print(f"  WARNING: Max iterations reached.")
    return True


def descend_to_place(target: str):
    """Descend while keeping centered on place target.

    Uses 3D detection (depth) to determine when to release:
    - Release when target is centered AND depth < PLACE_DEPTH_THRESHOLD
    - Falls back to PLACE_Z height if depth is unavailable
    Mask-only detection: as long as mask is visible, keep tracking.
    Monitors gripper object_detected state to catch premature drops.
    """
    ee_x, ee_y, ee_z = sensors.get_ee_position()
    print(f"\n--- Descend-to-Place: depth threshold={PLACE_DEPTH_THRESHOLD}m, "
          f"fallback Z={PLACE_Z:.3f}m ---")
    print(f"  Current EE Z: {ee_z:.3f}m")

    consecutive_misses = 0
    PLACE_HEIGHT_THRESHOLD = 0.02

    for i in range(MAX_SERVO_ITERATIONS):
        ee_x, ee_y, ee_z = sensors.get_ee_position()
        remaining = ee_z - PLACE_Z

        # Check if object was dropped prematurely during descent
        if not sensors.is_gripper_holding():
            print(f"  WARNING: Object lost during descent (gripper no longer holding)! "
                  f"Z={ee_z:.3f}m")
            return "dropped"

        # Fallback: if we've reached the fixed Z height, release
        if remaining <= PLACE_HEIGHT_THRESHOLD:
            print(f"  Reached fallback place height (Z={ee_z:.3f}m). Ready to release!")
            return True

        # Use 3D detection for depth info
        det, shape = detect_object_3d(target)

        if det is None:
            consecutive_misses += 1
            print(f"  Iter {i+1}: mask not detected "
                  f"({consecutive_misses}/{PLACE_LOST_RETRIES})")
            if consecutive_misses >= PLACE_LOST_RETRIES:
                print("  WARNING: Lost target mask, placing at current position.")
                return True
            time.sleep(0.5)
            continue
        consecutive_misses = 0

        obj_u, obj_v = get_object_pixel_center(det)
        cx, cy = get_image_center(shape)
        u_err = obj_u - cx
        v_err = obj_v - cy
        error_mag = np.sqrt(u_err**2 + v_err**2)

        # Get depth info
        depth = det.depth_meters if hasattr(det, 'depth_meters') else float('nan')
        depth_valid = not math.isnan(depth)
        depth_str = f"depth={depth:.3f}m" if depth_valid else "depth=NaN"

        print(f"  Iter {i+1}: err=({u_err:.0f},{v_err:.0f}) |{error_mag:.0f}px| "
              f"[mask] Z={ee_z:.3f}m {depth_str} remain={remaining*100:.1f}cm")

        # Depth-based release: centered + close enough
        if depth_valid and depth < PLACE_DEPTH_THRESHOLD and error_mag < PLACE_PIXEL_TOLERANCE:
            print(f"  DEPTH RELEASE: target at {depth:.3f}m < {PLACE_DEPTH_THRESHOLD}m "
                  f"and centered ({error_mag:.0f}px). Ready to release!")
            return True

        dx_lat, dy_lat, _ = pixel_error_to_ee_delta(u_err, v_err)

        descend_this_step = 0.0
        if error_mag < DESCEND_PAUSE_PIXELS:
            descend_this_step = min(PLACE_DESCEND_STEP_M, max(remaining, 0))
        else:
            print(f"    Pausing descent (error {error_mag:.0f} > "
                  f"{DESCEND_PAUSE_PIXELS}px), centering first...")

        dx = dx_lat
        dy = dy_lat
        dz = -descend_this_step

        total_step = np.sqrt(dx**2 + dy**2 + dz**2)
        if total_step < MIN_LATERAL_STEP_M and descend_this_step == 0:
            dz = -min(PLACE_DESCEND_STEP_M, max(remaining, 0))
            dx, dy = 0.0, 0.0

        print(f"    Move: dx={dx*1000:.1f}mm dy={dy*1000:.1f}mm dz={dz*1000:.1f}mm")

        try:
            arm.move_delta(dx=dx, dy=dy, dz=dz, droll=0, dpitch=0, dyaw=0,
                           frame="base", duration=SERVO_MOVE_DURATION)
        except ArmError as e:
            print(f"  CONTACT: {e}")
            return True

        time.sleep(0.2)

    print(f"  WARNING: Max iterations reached. Placing here.")
    return True


# ============================================================================
# Main place pipeline
# ============================================================================

def place_object(target: str = PLACE_TARGET):
    """Place held object onto target using visual servoing.

    Checks gripper object state before starting. If not holding, aborts early.
    Goes home, tilts camera, detects target from height (sees past held object),
    servos above target, descends while tracking, releases.
    Monitors object state throughout to detect premature drops.
    """
    print(f"=== Place Object on '{target}' ===\n")

    # --- Pre-check: verify gripper is holding an object ---
    holding = sensors.is_gripper_holding()
    print(f"Pre-check: gripper holding = {holding}")
    if not holding:
        print("  ERROR: Gripper is not holding an object. Aborting place.")
        display.show_face("sad")
        display.show_text("Nothing to place!")
        return False

    # --- Go home (holding object high) ---
    print("Phase 1: Going home (holding object high)...")
    display.show_text(f"Finding {target}...")
    display.show_face("thinking")
    arm.go_home()
    time.sleep(0.5)

    # Recheck after moving home — object may have slipped
    if not sensors.is_gripper_holding():
        print("  WARNING: Object lost after go_home! Aborting place.")
        display.show_face("sad")
        display.show_text("Object dropped!")
        arm.go_home()
        return False

    # --- Tilt camera down ---
    print("Phase 2: Tilting camera down...")
    arm.move_delta(dpitch=CAMERA_TILT_RAD, frame="ee", duration=1.0)
    time.sleep(0.3)

    # --- Detect place target ---
    print(f"\nPhase 3: Looking for place target '{target}'...")
    det, shape = detect_object_mask_only(target)
    if det is None:
        print(f"  Place target mask not visible, aborting place.")
        print(f"  Opening gripper to drop object...")
        gripper.open()
        time.sleep(0.5)
        arm.go_home()
        display.show_face("sad")
        return False

    obj_u, obj_v = get_object_pixel_center(det)
    print(f"  Place target detected at pixel ({obj_u:.0f}, {obj_v:.0f}) [mask]")

    # --- Servo above place target ---
    print("\nPhase 4: Centering above place target...")
    display.show_text(f"Moving above {target}...")
    centered = servo_above_place(target)
    if not centered:
        print("WARNING: Could not center, placing at best position.")

    # --- Descend to place height ---
    print("\nPhase 5: Descending to place height...")
    display.show_text(f"Lowering to {target}...")
    descent_result = descend_to_place(target)

    if descent_result == "dropped":
        print("\nObject was dropped during descent!")
        display.show_face("sad")
        display.show_text("Object dropped during descent!")
        arm.go_home()
        time.sleep(0.5)
        return False

    # --- Release ---
    print("\nPhase 6: Releasing object...")
    display.show_text("Releasing...")
    gripper.open()
    time.sleep(0.5)

    # Verify object was released (gripper should no longer detect object)
    still_holding = sensors.is_gripper_holding()
    if still_holding:
        print("  WARNING: Gripper still reports object after release. "
              "Object may be stuck.")
        display.show_face("thinking")
        display.show_text("Object may be stuck...")
    else:
        print("  Object released! (confirmed by gripper state)")
        display.show_face("happy")
        display.show_text(f"Placed on {target}!")

    # --- Go home ---
    print("\nPhase 7: Going home...")
    arm.go_home()
    time.sleep(0.5)

    print(f"\n=== Place complete! ===")
    display.show_face("excited")
    return not still_holding


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__" or True:
    success = place_object(PLACE_TARGET)
    print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
