import cv2
import numpy as np
import time
import serial

# =========================================================
# 시리얼 (아두이노 메가) 설정
# =========================================================

SERIAL_PORT = "COM6"
SERIAL_BAUD = 115200

# =========================================================
# 주행 제어 파라미터
# =========================================================

# 기본 전진 속도. 90이 바닥에서 차를 못 미는 것 같으면 천천히 올려보기.
# (210~220에서 드라이버가 탔으니 절대 거기 근처로는 가지 말 것)
BASE_SPEED = 110

# 곡선/조향 클 때 감속 하한. 너무 낮으면 바닥에서 안 움직임.
MIN_SPEED = 90

MAX_STEER = 100
STEER_GAIN = 0.35
STEER_SIGN = 1.0
STEER_SMOOTH = 0.5
LOST_FRAMES_BEFORE_STOP = 8

# =========================================================
# 카메라 설정
# =========================================================

CAMERA_INDEX = 1
FRAME_WIDTH = 640
FRAME_HEIGHT = 360

# =========================================================
# 차선 검출 설정
# =========================================================

# --- HSV 흰색 필터 ---
# PDF의 색상 필터링 단계. 흰색 차선은 채도(S)가 낮고 밝기(V)가 높다는 성질을 이용한다.
# 실내 조명 때문에 차선이 끊기면 HSV_WHITE_V_MIN을 낮추고, 흰색 물체가 많으면 S_MAX를 낮춘다.
HSV_WHITE_S_MAX = 80
HSV_WHITE_V_MIN = 150

# --- top-hat (빛 반사 제거 핵심) ---
# 이 커널보다 '두꺼운' 밝은 영역(바닥 반사 덩어리)은 제거되고,
# 이보다 얇은 밝은 선(차선)만 남는다.
# 차선이 통째로 사라지면 값을 키우고(예: 35), 반사가 덜 지워지면 줄인다(예: 15).
TOPHAT_KERNEL_SIZE = 25

# top-hat 결과를 이진화하는 임계값. 차선이 잘 안 잡히면 낮추고, 잡티 많으면 높임.
TOPHAT_THRESHOLD = 30

# --- Canny Edge ---
# PDF의 edge 검출 단계. 밝은 차선 후보 중 기하학적 경계가 있는 픽셀을 더 신뢰한다.
CANNY_LOW_THRESHOLD = 50
CANNY_HIGH_THRESHOLD = 150

# --- 컨투어 크기 필터 ---
MIN_CONTOUR_AREA = 250
MAX_LANE_PARTS = 4

# --- 모양 필터 (반사 덩어리 제거 핵심) ---
# 차선은 '가늘고 길다'. 길이/폭 비율이 이 값 이상인 것만 차선으로 인정.
# 반사 덩어리는 뭉툭해서 비율이 낮아 걸러진다.
# 너무 빡세서 차선까지 지워지면 1.8 정도로 낮추기.
MIN_ELONGATION = 2.2

# --- 차선 전용 기하 필터 ---
# 흰색 물체가 같이 잡히는 문제를 줄이기 위해, "길고 얇은 흰색"뿐 아니라
# 카메라 아래쪽 바닥 ROI에서 수직/대각선으로 이어지는 조각만 차선 후보로 본다.
MIN_CONTOUR_HEIGHT = 28
MAX_CONTOUR_WIDTH_RATIO = 0.30
MIN_CONTOUR_BOTTOM_Y = 0.55
MAX_ANGLE_FROM_VERTICAL = 65.0

# --- Bird's Eye View ---
# PDF의 원근 변환 단계. 실제 카메라 설치각에 따라 아래 source 점은 현장에서 보정해야 한다.
BIRD_SRC_BOTTOM_LEFT_X = 0.05
BIRD_SRC_BOTTOM_RIGHT_X = 0.95
BIRD_SRC_TOP_LEFT_X = 0.28
BIRD_SRC_TOP_RIGHT_X = 0.72
BIRD_SRC_TOP_Y = 0.58
BIRD_DST_LEFT_X = 0.25
BIRD_DST_RIGHT_X = 0.75

# --- Sliding Window ---
# PDF의 sliding window 기반 추적 단계. BEV 마스크에서 차선 픽셀 중심을 아래에서 위로 추적한다.
SLIDING_WINDOWS = 9
SLIDING_MARGIN = 55
SLIDING_MIN_PIXELS = 25
SLIDING_MIN_HISTOGRAM = 2000
SLIDING_MIN_FIT_PIXELS = 70
SLIDING_LOOKAHEAD_Y = 0.65
MIN_LANE_WIDTH_RATIO = 0.25
MAX_LANE_WIDTH_RATIO = 0.85
SINGLE_LANE_CENTER_OFFSET_RATIO = 0.24

# =========================================================
# ROI 설정
# =========================================================

ROI_BOTTOM_LEFT_X = 0
ROI_BOTTOM_RIGHT_X = 1
ROI_TOP_LEFT_X = 0.1
ROI_TOP_RIGHT_X = 0.9
ROI_TOP_Y = 0.6


def make_roi_mask(frame_shape):
    h, w = frame_shape[:2]
    roi_points = np.array([
        [
            (int(w * ROI_BOTTOM_LEFT_X), h),
            (int(w * ROI_BOTTOM_RIGHT_X), h),
            (int(w * ROI_TOP_RIGHT_X), int(h * ROI_TOP_Y)),
            (int(w * ROI_TOP_LEFT_X), int(h * ROI_TOP_Y))
        ]
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, roi_points, 255)
    return mask, roi_points


def make_birds_eye_points(frame_shape):
    h, w = frame_shape[:2]
    src = np.float32([
        [w * BIRD_SRC_BOTTOM_LEFT_X, h],
        [w * BIRD_SRC_BOTTOM_RIGHT_X, h],
        [w * BIRD_SRC_TOP_RIGHT_X, h * BIRD_SRC_TOP_Y],
        [w * BIRD_SRC_TOP_LEFT_X, h * BIRD_SRC_TOP_Y],
    ])
    dst = np.float32([
        [w * BIRD_DST_LEFT_X, h],
        [w * BIRD_DST_RIGHT_X, h],
        [w * BIRD_DST_RIGHT_X, 0],
        [w * BIRD_DST_LEFT_X, 0],
    ])
    return src, dst


def detect_lane(frame):
    """
    PDF의 YOLO 제외 CV 파이프라인 반영:
    1. HSV 색상 필터링으로 흰색 차선 후보 분리
    2. Canny Edge + top-hat으로 얇은 차선 경계 강조
    3. ROI 적용
    4. Bird's Eye View 원근 변환
    5. 작은 노이즈 정리
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, HSV_WHITE_V_MIN], dtype=np.uint8),
        np.array([179, HSV_WHITE_S_MAX, 255], dtype=np.uint8),
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    th_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (TOPHAT_KERNEL_SIZE, TOPHAT_KERNEL_SIZE)
    )
    tophat = cv2.morphologyEx(blurred, cv2.MORPH_TOPHAT, th_kernel)
    _, tophat_mask = cv2.threshold(
        tophat, TOPHAT_THRESHOLD, 255, cv2.THRESH_BINARY
    )

    edges = cv2.Canny(blurred, CANNY_LOW_THRESHOLD, CANNY_HIGH_THRESHOLD)
    edge_kernel = np.ones((3, 3), np.uint8)
    edge_mask = cv2.dilate(edges, edge_kernel, iterations=1)

    thin_white_mask = cv2.bitwise_and(
        white_mask,
        cv2.bitwise_or(tophat_mask, edge_mask),
    )

    roi_mask, roi_points = make_roi_mask(frame.shape)
    lane_mask = cv2.bitwise_and(thin_white_mask, roi_mask)

    kernel = np.ones((3, 3), np.uint8)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    src, dst = make_birds_eye_points(frame.shape)
    perspective = cv2.getPerspectiveTransform(src, dst)
    inverse_perspective = cv2.getPerspectiveTransform(dst, src)
    warped_mask = cv2.warpPerspective(
        lane_mask,
        perspective,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    warped_mask = cv2.morphologyEx(warped_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return warped_mask, roi_points, inverse_perspective


def mask_from_contours(mask_shape, lane_contours):
    filtered = np.zeros(mask_shape, dtype=np.uint8)
    if lane_contours:
        cv2.drawContours(filtered, lane_contours, -1, 255, thickness=cv2.FILLED)
    return filtered


def select_lane_contours(lane_mask):
    """
    크기 + 모양(가늘고 긴 것만)으로 컨투어를 거른다.
    반사 덩어리는 뭉툭해서 elongation 필터에서 떨어진다.
    """
    contours, _ = cv2.findContours(
        lane_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask_height, mask_width = lane_mask.shape[:2]
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if h < MIN_CONTOUR_HEIGHT:
            continue
        if w > mask_width * MAX_CONTOUR_WIDTH_RATIO:
            continue
        if y + h < mask_height * MIN_CONTOUR_BOTTOM_Y:
            continue

        # 최소 회전 사각형으로 진짜 길쭉함 측정 (기울어진 차선도 OK)
        (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side = max(rw, rh)
        short_side = min(rw, rh)
        if short_side < 1:
            continue

        elongation = long_side / short_side
        if elongation < MIN_ELONGATION:
            # 뭉툭한 반사 덩어리 -> 버림
            continue

        # 가로에 가까운 흰색 물체/표식은 주행 차선으로 쓰지 않는다.
        line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        vx, vy = float(line[0]), float(line[1])
        raw_angle = abs(np.degrees(np.arctan2(vx, vy)))
        angle_from_vertical = min(raw_angle, abs(180.0 - raw_angle))
        if angle_from_vertical > MAX_ANGLE_FROM_VERTICAL:
            continue

        valid_contours.append(contour)

    valid_contours.sort(
        key=lambda c: (
            cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3],
            cv2.contourArea(c),
        ),
        reverse=True,
    )
    return valid_contours[:MAX_LANE_PARTS]


def _histogram_base_x(histogram, x_offset):
    if histogram.size == 0:
        return None
    peak_index = int(np.argmax(histogram))
    if histogram[peak_index] < SLIDING_MIN_HISTOGRAM:
        return None
    return peak_index + x_offset


def _collect_sliding_window_pixels(lane_mask, base_x):
    height = lane_mask.shape[0]
    window_height = height // SLIDING_WINDOWS
    nonzero_y, nonzero_x = lane_mask.nonzero()

    current_x = int(base_x)
    lane_indices = []
    for window in range(SLIDING_WINDOWS):
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        win_x_low = current_x - SLIDING_MARGIN
        win_x_high = current_x + SLIDING_MARGIN

        good_indices = (
            (nonzero_y >= win_y_low)
            & (nonzero_y < win_y_high)
            & (nonzero_x >= win_x_low)
            & (nonzero_x < win_x_high)
        ).nonzero()[0]
        lane_indices.append(good_indices)

        if len(good_indices) >= SLIDING_MIN_PIXELS:
            current_x = int(np.mean(nonzero_x[good_indices]))

    if not lane_indices:
        return np.array([]), np.array([])

    lane_indices = np.concatenate(lane_indices)
    return nonzero_x[lane_indices], nonzero_y[lane_indices]


def _fit_lane_x_at_y(xs, ys, y_eval):
    if len(xs) < SLIDING_MIN_FIT_PIXELS:
        return None
    fit = np.polyfit(ys, xs, 2)
    return float(fit[0] * y_eval * y_eval + fit[1] * y_eval + fit[2])


def _infer_center_from_single_lane(lane_x, width, is_left_lane):
    offset = width * SINGLE_LANE_CENTER_OFFSET_RATIO
    if is_left_lane:
        return lane_x + offset
    return lane_x - offset


def compute_lane_center_x(lane_mask):
    if cv2.countNonZero(lane_mask) == 0:
        return None

    height, width = lane_mask.shape[:2]
    histogram_start_y = int(height * 0.55)
    histogram = np.sum(lane_mask[histogram_start_y:, :], axis=0)
    midpoint = width // 2

    left_base_x = _histogram_base_x(histogram[:midpoint], 0)
    right_base_x = _histogram_base_x(histogram[midpoint:], midpoint)
    if left_base_x is None and right_base_x is None:
        return None

    y_eval = int(height * SLIDING_LOOKAHEAD_Y)
    left_x = right_x = None
    left_count = right_count = 0

    if left_base_x is not None:
        xs, ys = _collect_sliding_window_pixels(lane_mask, left_base_x)
        left_count = len(xs)
        left_x = _fit_lane_x_at_y(xs, ys, y_eval)

    if right_base_x is not None:
        xs, ys = _collect_sliding_window_pixels(lane_mask, right_base_x)
        right_count = len(xs)
        right_x = _fit_lane_x_at_y(xs, ys, y_eval)

    min_lane_width = width * MIN_LANE_WIDTH_RATIO
    max_lane_width = width * MAX_LANE_WIDTH_RATIO
    lane_center_x = None

    if left_x is not None and right_x is not None:
        lane_width = abs(right_x - left_x)
        if min_lane_width <= lane_width <= max_lane_width:
            lane_center_x = (left_x + right_x) / 2.0
        elif left_count >= right_count:
            lane_center_x = _infer_center_from_single_lane(left_x, width, True)
        else:
            lane_center_x = _infer_center_from_single_lane(right_x, width, False)
    elif left_x is not None:
        lane_center_x = _infer_center_from_single_lane(left_x, width, True)
    elif right_x is not None:
        lane_center_x = _infer_center_from_single_lane(right_x, width, False)

    if lane_center_x is None:
        return None
    return float(max(0.0, min(width - 1.0, lane_center_x)))


def project_birds_eye_point(x, y, inverse_perspective):
    point = np.array([[[float(x), float(y)]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, inverse_perspective)[0][0]
    return int(round(projected[0])), int(round(projected[1]))


# =========================================================
# 시리얼 헬퍼
# =========================================================

def open_serial():
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    time.sleep(2.0)
    start = time.time()
    while time.time() - start < 3.0:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print("[ARDUINO]", line)
        if "READY" in line:
            break
    return ser


def send_drive(ser, speed, steer):
    speed = int(max(0, min(255, speed)))
    steer = int(max(-MAX_STEER, min(MAX_STEER, steer)))
    ser.write(f"DRIVE {speed} {steer}\n".encode())


def send_stop(ser):
    ser.write(b"STOP\n")


# =========================================================
# 메인 주행 루프
# =========================================================

def main():
    try:
        ser = open_serial()
        print("시리얼 연결 완료:", SERIAL_PORT)
    except Exception as e:
        print("시리얼 연결 실패:", e)
        print("COM6가 맞는지, 시리얼 모니터/다른 프로그램이 점유 중인지 확인하세요.")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("카메라를 열 수 없습니다. CAMERA_INDEX를 0, 1, 2로 바꿔보세요.")
        send_stop(ser)
        ser.close()
        return

    prev_time = time.time()
    smoothed_steer = 0.0
    lost_count = 0
    driving = False
    print("스페이스바: 주행 시작/정지 토글  |  q: 종료")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽지 못했습니다.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            raw_lane_mask, roi_points, inverse_perspective = detect_lane(frame)
            lane_contours = select_lane_contours(raw_lane_mask)
            lane_mask = mask_from_contours(raw_lane_mask.shape, lane_contours)
            lane_center_x = compute_lane_center_x(lane_mask)

            result = frame.copy()
            cv2.polylines(result, roi_points, True, (255, 0, 0), 2)
            original_lane_mask = cv2.warpPerspective(
                lane_mask,
                inverse_perspective,
                (FRAME_WIDTH, FRAME_HEIGHT),
                flags=cv2.INTER_NEAREST,
            )
            lane_pixels = original_lane_mask > 0
            result[lane_pixels] = (
                result[lane_pixels] * 0.45
                + np.array([0, 255, 0], dtype=np.uint8) * 0.55
            ).astype(np.uint8)

            frame_center_x = FRAME_WIDTH / 2.0

            if lane_center_x is not None:
                lost_count = 0
                error = lane_center_x - frame_center_x
                raw_steer = STEER_SIGN * STEER_GAIN * error
                smoothed_steer = (
                    STEER_SMOOTH * smoothed_steer
                    + (1 - STEER_SMOOTH) * raw_steer
                )
                steer_cmd = int(max(-MAX_STEER, min(MAX_STEER, smoothed_steer)))
                speed_cmd = int(
                    BASE_SPEED
                    - (BASE_SPEED - MIN_SPEED) * (abs(steer_cmd) / MAX_STEER)
                )

                lane_bottom = project_birds_eye_point(
                    lane_center_x, FRAME_HEIGHT - 1, inverse_perspective
                )
                lane_lookahead = project_birds_eye_point(
                    lane_center_x,
                    int(FRAME_HEIGHT * SLIDING_LOOKAHEAD_Y),
                    inverse_perspective,
                )
                frame_bottom = project_birds_eye_point(
                    frame_center_x, FRAME_HEIGHT - 1, inverse_perspective
                )
                frame_lookahead = project_birds_eye_point(
                    frame_center_x,
                    int(FRAME_HEIGHT * SLIDING_LOOKAHEAD_Y),
                    inverse_perspective,
                )
                cv2.line(result, lane_bottom, lane_lookahead, (0, 0, 255), 2)
                cv2.line(result, frame_bottom, frame_lookahead, (255, 255, 0), 1)

                if driving:
                    send_drive(ser, speed_cmd, steer_cmd)
                status = f"DRIVE s={speed_cmd} st={steer_cmd}"
            else:
                lost_count += 1
                if driving and lost_count >= LOST_FRAMES_BEFORE_STOP:
                    send_stop(ser)
                    status = "LANE LOST -> STOP"
                else:
                    status = f"LANE LOST ({lost_count})"

            mode = "RUN" if driving else "PAUSE"
            color = (0, 255, 0) if driving else (0, 0, 255)
            cv2.putText(result, mode, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(result, status, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            now = time.time()
            fps = 1.0 / (now - prev_time) if now > prev_time else 0.0
            prev_time = now
            cv2.putText(result, f"FPS: {fps:.1f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Drive", result)
            cv2.imshow("BEV Lane Mask", lane_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                driving = not driving
                if not driving:
                    send_stop(ser)
                    smoothed_steer = 0.0
                print("주행:", "ON" if driving else "OFF")

    finally:
        send_stop(ser)
        time.sleep(0.2)
        ser.close()
        cap.release()
        cv2.destroyAllWindows()
        print("정지 명령 전송 후 종료.")


if __name__ == "__main__":
    main()
