# Architecture

```text
OpenCV camera frame
        |
        v
camera/camera_source.py
        |
        v
recognition/opencv_lane.py
        |
        v
decision/lane_follower.py
        |
        v
control/protocol.py + control/arduino_serial.py
        |
        v
Arduino motor controller
```

원칙:

- `camera`: 입력 장치만 다룬다.
- `recognition`: 이미지에서 차선/객체를 찾는다.
- `decision`: 인식 결과를 조향/속도로 바꾼다.
- `control`: 명령을 하드웨어에 보낸다.

이렇게 나누면 YOLO를 추가할 때 `recognition/`만 교체하고, `decision/`과 `control/`은 그대로 쓸 수 있습니다.
