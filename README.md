# SKKU Autocar

Mac에서 바로 실험할 수 있도록 ROS 없이 다시 구성한 자율주행 프로젝트입니다.

전체 흐름은 네 단계입니다.

```text
camera       -> 카메라 프레임 입력
recognition  -> 차선/객체 인식
decision     -> 조향/속도 판단
control      -> Arduino로 명령 전송
```

## Folder Roles

자세한 역할 설명은 [docs/folder_roles.md](docs/folder_roles.md)에 정리했습니다.

핵심 폴더:

- `src/autocar/camera/`: 카메라 열기, 프레임 읽기
- `src/autocar/recognition/`: 프레임에서 차선 정보 추출
- `src/autocar/decision/`: 차선 정보로 주행 명령 계산
- `src/autocar/control/`: 주행 명령을 Arduino serial 명령으로 변환/전송
- `scripts/`: 실제 실행 스크립트
- `config/`: 카메라, ROI, 속도, 포트 설정
- `arduino/`: Arduino에 업로드할 제어 코드

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Camera Probe

폰 카메라나 내장 카메라 번호를 확인합니다.

```bash
python3 scripts/probe_cameras.py
```

## Baseline Preview

Arduino 없이 카메라/차선/판단만 확인합니다.

```bash
python3 scripts/run_baseline.py --config config/default.json --camera-index 0
```

폰 카메라가 1번이면:

```bash
python3 scripts/run_baseline.py --config config/default.json --camera-index 1
```

키:

- `space`: 판단 시작/일시정지
- `s`: 정지
- `q`: 종료

## Real Drive

먼저 바퀴를 띄운 상태에서 테스트합니다.

```bash
python3 scripts/run_baseline.py \
  --config config/default.json \
  --camera-index 1 \
  --serial-port /dev/cu.usbmodemXXXX \
  --drive
```

차가 반대로 꺾이면 `config/default.json`의 `decision.steering_sign`을 `-1`로 바꾸면 됩니다.

## Arduino

[arduino/vehicle_controller/vehicle_controller.ino](arduino/vehicle_controller/vehicle_controller.ino)를 Arduino Mega에 업로드합니다.

현재 코드의 배선 기준:

```text
오른쪽 뒷바퀴: IN1=D3,  IN2=D4
왼쪽 뒷바퀴:   IN1=D7,  IN2=D8
핸들:          IN1=D11, IN2=D12
```

Python 쪽에서 Arduino로 보내는 명령 형식:

```text
DRIVE <speed> <steering>
STOP
PING
```

예:

```text
DRIVE 80 -20
STOP
```

업로드 후 바퀴를 띄운 상태에서 serial 테스트:

```bash
python3 scripts/test_serial.py --port /dev/cu.usbmodem14101
```

구동 바퀴까지 아주 낮은 속도로 테스트:

```bash
python3 scripts/test_serial.py --port /dev/cu.usbmodem14101 --speed 45
```

현재 구동 테스트에서 양쪽 뒷바퀴가 뒤로 돌아 `vehicle_controller.ino`의 `RIGHT_DRIVE_INVERT`, `LEFT_DRIVE_INVERT`는 `true`로 설정되어 있습니다. 한쪽 바퀴만 반대로 돌면 해당 값만 다시 조정하세요. 핸들이 반대로 움직이면 `STEERING_INVERT`를 `true`로 바꿉니다.
