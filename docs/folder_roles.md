# Folder Roles

이 프로젝트는 역할을 섞지 않기 위해 네 계층으로 나눕니다.

## `src/autocar/camera/`

역할: 카메라 입력만 담당합니다.

하는 일:

- OpenCV `VideoCapture` 열기
- 카메라 번호, 해상도, FPS 설정
- 프레임 읽기

하지 않는 일:

- 차선 판단
- 조향 계산
- Arduino 명령 전송

대표 파일:

- `camera_source.py`: 실제 카메라에서 프레임을 읽는 클래스

## `src/autocar/recognition/`

역할: 이미지에서 의미 있는 정보를 뽑습니다.

현재 하는 일:

- ROI 설정
- grayscale threshold 기반 흰 차선 검출
- 차선 중심 offset 계산
- 디버그 화면용 overlay 생성

나중에 추가할 일:

- YOLO 기반 차선/신호등/장애물 인식
- 라이다 장애물 인식 결과와 결합

하지 않는 일:

- 속도 결정
- 조향 결정
- Arduino 전송

대표 파일:

- `opencv_lane.py`: YOLO 없이 동작하는 baseline 차선 인식기

## `src/autocar/decision/`

역할: 인식 결과를 보고 어떻게 움직일지 판단합니다.

하는 일:

- 차선 중심이 화면 중앙에서 얼마나 벗어났는지 확인
- 조향값 계산
- 기본 속도 결정
- 차선을 못 보면 정지 명령 생성

하지 않는 일:

- 카메라를 직접 열기
- OpenCV threshold 수행
- serial write 수행

대표 파일:

- `lane_follower.py`: 차선 추종 판단 로직

## `src/autocar/control/`

역할: 판단 결과를 실제 하드웨어 명령으로 바꿉니다.

하는 일:

- `DriveCommand`를 serial 문자열로 인코딩
- Arduino serial 연결
- `DRIVE speed steering` 또는 `STOP` 전송

하지 않는 일:

- 차선 인식
- 주행 판단
- 카메라 접근

대표 파일:

- `protocol.py`: 명령 문자열 변환
- `arduino_serial.py`: Arduino serial 연결/전송

## `src/autocar/common/`

역할: 여러 계층이 공유하는 자료형과 설정 로더를 둡니다.

대표 파일:

- `types.py`: `LaneInfo`, `DriveCommand` 같은 공통 자료형
- `config.py`: JSON 설정 파일 로더

## `scripts/`

역할: 사람이 직접 실행하는 진입점입니다.

대표 파일:

- `run_baseline.py`: camera -> recognition -> decision -> control 전체 루프
- `probe_cameras.py`: 사용 가능한 카메라 번호 확인
- `test_serial.py`: Arduino serial 명령 단독 테스트

## `config/`

역할: 코드 수정 없이 바꿀 수 있는 튜닝값을 둡니다.

예:

- 카메라 해상도
- ROI 위치
- threshold
- 기본 속도
- 조향 gain
- Arduino serial port

## `arduino/`

역할: Arduino에 업로드할 펌웨어를 둡니다.

Arduino는 판단하지 않습니다. 노트북에서 받은 명령대로 모터만 제어합니다.
