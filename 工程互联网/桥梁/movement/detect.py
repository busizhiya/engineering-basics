import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import pandas as pd
from pathlib import Path

TRACKING_SCALE = 0.7
UPDATE_EVERY_N_FRAMES = 2
DISPLAY_MAX_WIDTH = 960
PLOT_OUTPUT_PATH = "tracking_plot.png"
USE_SUBPIXEL = True
SENSOR_CSV_NAME = "displacement_wide.csv"


def plot_tracking_results(frame_ids, dx_values, dy_values):
    if not frame_ids:
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(frame_ids, dx_values, label="dx", color="tab:blue", linewidth=1.4)
    plt.plot(frame_ids, dy_values, label="dy", color="tab:orange", linewidth=1.4)
    plt.xlabel("Frame")
    plt.ylabel("Displacement (pixels)")
    plt.title("Displacement Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dx_values, dy_values, color="tab:green", linewidth=1.2)
    plt.scatter([dx_values[0]], [dy_values[0]], color="tab:blue", label="Start", zorder=3)
    plt.scatter([dx_values[-1]], [dy_values[-1]], color="tab:red", label="End", zorder=3)
    plt.xlabel("dx (pixels)")
    plt.ylabel("dy (pixels)")
    plt.title("Motion Trajectory")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH, dpi=160)
    plt.show()


def create_tracker():
    tracker_factory = getattr(cv2, "TrackerCSRT_create", None)
    if tracker_factory is None:
        legacy = getattr(cv2, "legacy", None)
        tracker_factory = getattr(legacy, "TrackerCSRT_create", None) if legacy is not None else None
    if tracker_factory is not None:
        return tracker_factory()
    return TemplateMatchTracker()


class TemplateMatchTracker:
    def __init__(self):
        self.template_gray = None
        self.bbox = None

    @staticmethod
    def _quadratic_offset(v_left, v_center, v_right):
        denom = v_left - 2.0 * v_center + v_right
        if abs(denom) < 1e-8:
            return 0.0
        offset = 0.5 * (v_left - v_right) / denom
        return float(max(-1.0, min(1.0, offset)))

    def _subpixel_peak(self, response, px, py):
        h, w = response.shape[:2]
        if px <= 0 or py <= 0 or px >= w - 1 or py >= h - 1:
            return 0.0, 0.0
        cx = float(response[py, px])
        left = float(response[py, px - 1])
        right = float(response[py, px + 1])
        top = float(response[py - 1, px])
        bottom = float(response[py + 1, px])
        dx = self._quadratic_offset(left, cx, right)
        dy = self._quadratic_offset(top, cx, bottom)
        return dx, dy

    def init(self, frame, bbox):
        x, y, w, h = map(int, bbox)
        if w <= 0 or h <= 0:
            raise ValueError("Invalid ROI selected")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template_gray = gray[y : y + h, x : x + w].copy()
        self.bbox = (x, y, w, h)

    def update(self, frame):
        if self.template_gray is None or self.bbox is None:
            return False, self.bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.bbox

        margin_x = max(20.0, w)
        margin_y = max(20.0, h)
        left = int(max(0, round(x - margin_x)))
        top = int(max(0, round(y - margin_y)))
        right = int(min(gray.shape[1], round(x + w + margin_x)))
        bottom = int(min(gray.shape[0], round(y + h + margin_y)))

        search_region = gray[top:bottom, left:right]
        template_h, template_w = self.template_gray.shape[:2]
        if search_region.shape[0] < template_h or search_region.shape[1] < template_w:
            return False, self.bbox

        match = cv2.matchTemplate(search_region, self.template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match)
        if max_val < 0.3:
            return False, self.bbox

        sub_dx, sub_dy = 0.0, 0.0
        if USE_SUBPIXEL:
            sub_dx, sub_dy = self._subpixel_peak(match, max_loc[0], max_loc[1])

        self.bbox = (left + max_loc[0] + sub_dx, top + max_loc[1] + sub_dy, template_w, template_h)
        return True, self.bbox


def run_tracking(video_path="input.mp4"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-6:
        fps = 30.0

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Could not read first frame from {video_path}")

    bbox = cv2.selectROI("Select Object", frame, False)
    cv2.destroyWindow("Select Object")
    if bbox == (0, 0, 0, 0):
        cap.release()
        cv2.destroyAllWindows()
        raise RuntimeError("No ROI selected")

    tracker = create_tracker()
    proc_frame = frame
    if TRACKING_SCALE != 1.0:
        proc_frame = cv2.resize(frame, None, fx=TRACKING_SCALE, fy=TRACKING_SCALE, interpolation=cv2.INTER_AREA)

    sx, sy, sw, sh = bbox
    tracker.init(proc_frame, (sx * TRACKING_SCALE, sy * TRACKING_SCALE, sw * TRACKING_SCALE, sh * TRACKING_SCALE))

    x, y, w, h = map(float, bbox)
    x0, y0 = x + w / 2, y + h / 2

    frame_idx = 0
    last_bbox = (x, y, w, h)
    last_success = True
    tracked_frames, dx_history, dy_history = [], [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        should_update = frame_idx % UPDATE_EVERY_N_FRAMES == 0

        if should_update:
            proc_frame = frame
            if TRACKING_SCALE != 1.0:
                proc_frame = cv2.resize(frame, None, fx=TRACKING_SCALE, fy=TRACKING_SCALE, interpolation=cv2.INTER_AREA)
            success, proc_bbox = tracker.update(proc_frame)
            if success and proc_bbox is not None:
                px, py, pw, ph = proc_bbox
                bbox = (px / TRACKING_SCALE, py / TRACKING_SCALE, pw / TRACKING_SCALE, ph / TRACKING_SCALE)
                last_bbox = bbox
            last_success = success
        else:
            success = last_success
            bbox = last_bbox

        if success and bbox is not None:
            x, y, w, h = bbox
            cx, cy = x + w / 2, y + h / 2
            dx, dy = cx - x0, cy - y0
            tracked_frames.append(frame_idx)
            dx_history.append(dx)
            dy_history.append(dy)

            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            cv2.putText(frame, f"dx={dx:.3f}, dy={dy:.3f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        display_frame = frame
        frame_h, frame_w = frame.shape[:2]
        if frame_w > DISPLAY_MAX_WIDTH:
            scale = DISPLAY_MAX_WIDTH / frame_w
            display_frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, int(frame_h * scale)), interpolation=cv2.INTER_AREA)

        cv2.imshow("Tracking", display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    plot_tracking_results(tracked_frames, dx_history, dy_history)

    tracking_df = pd.DataFrame({"frame_id": tracked_frames, "dx_pixels": dx_history, "dy_pixels": dy_history})
    tracking_df["time"] = tracking_df["frame_id"] / float(fps)
    tracking_df.to_csv("tracking_result.csv", index=False)
    print("✓ Tracking result saved to: tracking_result.csv")
    return tracking_df


def launch_manual_alignment_ui(tracking_df):
    csv_path = Path(__file__).parent.parent / SENSOR_CSV_NAME
    if not csv_path.exists():
        print(f"✗ Missing sensor CSV: {csv_path}")
        return

    sensor_df = pd.read_csv(csv_path)
    required = {"time", "dist_01.dist"}
    if not required.issubset(sensor_df.columns):
        raise KeyError("displacement_wide.csv must contain columns: time, dist_01.dist")

    t = tracking_df["time"].to_numpy(dtype=float)
    dy = tracking_df["dy_pixels"].to_numpy(dtype=float)
    st = sensor_df["time"].to_numpy(dtype=float)
    sd = sensor_df["dist_01.dist"].to_numpy(dtype=float)

    # Normalize dist into [182, 190] and invert direction: low->190, high->182.
    sd_min = float(np.min(sd))
    sd_max = float(np.max(sd))
    if sd_max - sd_min < 1e-8:
        raise RuntimeError("dist_01.dist has near-zero variance; cannot map to 182-190")
    sd_norm = (sd - sd_min) / (sd_max - sd_min)
    sd_mapped = 190.0 - 8.0 * sd_norm

    fig = plt.figure(figsize=(13, 8))
    ax_main = fig.add_axes([0.08, 0.36, 0.88, 0.58])
    ax_err = fig.add_axes([0.08, 0.18, 0.88, 0.12], sharex=ax_main)
    ax_shift = fig.add_axes([0.08, 0.12, 0.62, 0.03])
    ax_scale = fig.add_axes([0.08, 0.08, 0.62, 0.03])
    ax_offset = fig.add_axes([0.08, 0.04, 0.62, 0.03])
    ax_save = fig.add_axes([0.76, 0.05, 0.18, 0.08])

    s_shift = Slider(ax_shift, "Time Shift(s)", -2.0, 2.0, valinit=0.0, valstep=0.005)
    s_scale = Slider(ax_scale, "DY Scale", -5.0, 5.0, valinit=1.0, valstep=0.001)
    s_offset = Slider(ax_offset, "DY Offset", 180.0, 200.0, valinit=0.0, valstep=0.1)
    btn_save = Button(ax_save, "Save CSV")

    line_dist, = ax_main.plot(t, np.zeros_like(dy), color="tab:red", linewidth=1.8, label="dist mapped (182-190, inverted)")
    line_dy, = ax_main.plot(t, np.zeros_like(dy), color="tab:blue", linewidth=1.5, label="dy adjusted")
    line_err, = ax_err.plot(t, np.zeros_like(dy), color="tab:green", linewidth=1.1)
    ax_main.grid(True, alpha=0.3)
    ax_err.grid(True, alpha=0.3)
    ax_main.legend(loc="best")
    ax_main.set_ylabel("Displacement")
    ax_err.set_ylabel("Residual")
    ax_err.set_xlabel("Time (s)")

    state = {"dist_mapped": None, "dy_adjusted": None, "residual": None, "metrics": None}

    def refresh(_=None):
        shift = float(s_shift.val)
        scale = float(s_scale.val)
        offset = float(s_offset.val)
        dist_mapped = np.interp(t - shift, st, sd_mapped)
        dy_adjusted = scale * dy + offset
        residual = dy_adjusted - dist_mapped
        rmse = float(np.sqrt(np.mean(residual**2)))
        mae = float(np.mean(np.abs(residual)))
        corr = float(np.corrcoef(dy_adjusted, dist_mapped)[0, 1]) if np.std(dy_adjusted) > 1e-8 and np.std(dist_mapped) > 1e-8 else np.nan

        line_dist.set_ydata(dist_mapped)
        line_dy.set_ydata(dy_adjusted)
        line_err.set_ydata(residual)
        ax_main.set_ylim(181.5, 190.5)
        ax_main.set_title(
            f"Manual dist-dy alignment | dist fixed in [182,190] inverted | shift={shift:.3f}s dy_scale={scale:.3f} dy_offset={offset:.2f} corr={corr:.4f}"
        )
        ax_err.set_title(f"RMSE={rmse:.4f} MAE={mae:.4f}")
        fig.canvas.draw_idle()

        state["dist_mapped"] = dist_mapped
        state["dy_adjusted"] = dy_adjusted
        state["residual"] = residual
        state["metrics"] = {
            "time_shift_sec": shift,
            "dy_scale": scale,
            "dy_offset": offset,
            "corr": corr,
            "rmse": rmse,
            "mae": mae,
        }

    def save_current(_event):
        if state["dy_adjusted"] is None:
            refresh()
        out_df = pd.DataFrame(
            {
                "frame_id": tracking_df["frame_id"],
                "time": t,
                "dy_pixels": dy,
                "dy_adjusted": state["dy_adjusted"],
                "dist_mapped_182_190_inverted": state["dist_mapped"],
                "diff_dy_adjusted_minus_dist_mapped": state["residual"],
            }
        )
        out_df.to_csv("tracking_dist_dy_aligned_manual.csv", index=False)
        pd.DataFrame([state["metrics"]]).to_csv("tracking_dist_dy_manual_params.csv", index=False)
        print("✓ Saved: tracking_dist_dy_aligned_manual.csv")
        print("✓ Saved: tracking_dist_dy_manual_params.csv")

    s_shift.on_changed(refresh)
    s_scale.on_changed(refresh)
    s_offset.on_changed(refresh)
    btn_save.on_clicked(save_current)
    refresh()
    plt.show()


def main():
    tracking_df = run_tracking("input.mp4")
    launch_manual_alignment_ui(tracking_df)


if __name__ == "__main__":
    main()
