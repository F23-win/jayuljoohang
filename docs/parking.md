# T자 주차 런타임

`scripts/parking.py`는 일반 차선 주행과 분리된 주차 전용 런타임이다. 다음
순서를 그대로 상태 머신으로 실행한다.

1. 라이다로 주차칸 양옆의 주차 차량 두 대를 찾는다.
2. 두 차량 표면 사이 간격의 중심에 차량 후축을 맞춘다.
3. 두 차량 사이에 공식 크기 `950 × 1500 mm`의 가상 주차칸을 만든다.
4. 라이다 주차칸 중심과 가상 뒷선을 향하는 후진 경로를 만들고 추종한다.
5. 차량이 가상 주차칸 안에 들어오고 뒷선 여유 거리에 도달하면 정지한다.

대회 도면상 주차칸 크기는 `950 × 1500 mm`이다. 도로 폭 `850 mm`는 주차
제어에 사용하지 않는다. 라이다는 페인트 선을 보는 센서가 아니므로,
`parking_space_width_mm=950`과 실제 로그에서 측정한 이웃 차량 표면 간격
`expected_observed_gap_mm=1375`를 별도 설정값으로 유지한다.

## 모델

주차 모델은 일반 주행 모델과 분리한다.

```text
trained_model/parking_best.pt
```

Roboflow COCO segmentation 데이터로 `yolov8n-seg.pt`를 미세조정한 모델이며,
ㄷ자의 세 직선은 각각 별도 `line` 인스턴스여야 한다.

## 녹화 ZIP 재생

ZIP 안에 MP4 한 개와 `*_lidar.csv` 한 개가 있으면 직접 재생할 수 있다.
녹화 재생에서는 시리얼 출력이 강제로 금지된다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --device cpu `
  --imgsz 512 `
  --frame-stride 2 `
  --auto-start
```

CPU에서 mask가 불안정하면 `--imgsz 640 --frame-stride 1`로 되돌린다. 녹화
재생의 상태 전이와 라이다 조회는 벽시계 시간이 아니라 영상 시간으로
동기화된다.

조작키:

- `Space`: 미션 시작/취소
- `R`: 라이다·카메라 추정기와 상태 머신 초기화
- `Q` 또는 `Esc`: 종료

## 실시간 통합 화면과 자동 녹화

숫자 카메라 소스로 실행하면 녹화 재생과 동일한 `1280 × 720` 통합 화면 하나를
표시한다. 왼쪽은 후방카메라와 YOLO mask, 오른쪽 위는 BEV, 오른쪽 아래는
LiDAR이며 아래쪽에는 상태 머신·출력 명령·LiDAR 상태가 표시된다. 화면에 표시한
동일 프레임은 별도 옵션 없이 자동으로 다음 경로에 저장된다.

```text
data/parking/YYYYMMDD_HHMMSS.mp4
```

macOS 실시간 실행 예시(포트명은 실제 장치명으로 교체):

```bash
python3 scripts/parking.py \
  --source 1 \
  --device mps \
  --lidar-port /dev/tty.usbserial-LIDAR \
  --serial \
  --serial-port /dev/tty.usbmodem-ARDUINO
```

`--record-dashboard off`로 자동 녹화를 끌 수 있고, 녹화 영상 재생에서도 새
대시보드 영상을 만들려면 `--record-dashboard on`을 사용한다. 저장 폴더와 FPS는
각각 `--parking-record-dir`, `--dashboard-record-fps`로 바꾼다. `Q` 또는 `Esc`로
종료하면 모터 정지 명령을 보낸 뒤 MP4를 정상적으로 닫는다.

## 디버그 화면

후방카메라/BEV mask 색상:

- 청록: 왼쪽 주차선
- 초록: 오른쪽 주차선
- 빨강: 주차 뒷선
- 자홍: 선택된 주차칸에 포함되지 않은 line 인스턴스
- 주황 곡선: 생성된 후진 경로
- 노랑 점: 현재 look-ahead 목표
- 흰 원: 뒷선 앞 최종 정지 목표

라이다 화면:

- 주황 사각형: 감지된 두 이웃 차량의 라이다 방향 표면에서 시작해 주차장
  안쪽으로 1500mm만 확장되는 주차 공간. 장애물 중심을 기준으로 앞뒤 절반씩
  확장하지 않는다.
- 빨강 사각형: 긴급 정지용 좁은 직후방 ROI
- 파랑 사각형: 주차 차량 군집. 주황 공간의 양쪽 기준점이 된다.
- 첫 차량을 2개 고유 scan에서 확인한 직후 차량 군집 ROI를
  `xRight=-1800~2600mm`, `yBack=-2500~2500mm`로 고정 전환한다. 좌조향 중
  처음 오른쪽에 있던 차량이 왼쪽 좌표로 넘어가도 좁은 초기 ROI로 돌아가지
  않는다. `-1800mm`는 라이다 반경이 아니라 차량 좌표계의 왼쪽 경계이며,
  라이다 원시 거리 상한은 별도로 `12000mm`이다.
- 확장 ROI는 이미 확인한 첫 차량을 놓치지 않고 추적하는 용도다. 확장 ROI에
  들어온 임의의 두 군집을 주차 차량으로 연결하지 않는다.
- 좌조향 전진 중 첫 차량이 후축 기준 게이트를 지나면 두 번째 차량 검출을
  ARM한다. 이후 `xRight=800~2600mm`, 후축 기준
  `y=-700~500mm`인 우측 후방 게이트에 새로 들어온 군집만 두 번째 차량
  후보로 사용한다. 이 후보와 추적 중인 첫 차량의 표면 간격이
  `1100~1650mm`일 때 고유 scan 한 번으로 주차칸을 확정한다.
- 검출된 표면 간격을 주차칸 폭으로 쓰지는 않고, 두 차량의 중심과 방향에
  공식 크기 `950 × 1500mm` 박스를 만든다.
- 최초 확정 뒤에는 같은 두 차량과 주차칸 깊이 방향을 추적한다. 중심과 각도는
  새 관측에 따라 계속 갱신하지만 이전 방향과 반대인 180도 후보 및 35도보다
  큰 단일 스캔 점프는 거부한다. 두 차량을 모두 놓친 동안에는 마지막 박스를
  `HOLD`로 유지하며, 이 값으로 POSITIONING 상태를 진행하지는 않는다.
- 초록 선: 두 차량을 기준으로 만든 동적 주차 공간의 중앙선
- 청록 짧은 선: 라이다보다 앞쪽에 있는 차량 후축 기준선
- 연두 점: 라이다 측정점. 안전 ROI 안에서도 빨간 점으로 바꾸지 않는다.
- 흰색 박스와 화살표: `550 × 1000mm` 차량 외곽선과 차량 앞 방향. 라이다를
  뒤 범퍼보다 100mm 뒤에 둔 임시 장착값을 사용하므로 박스 전체가 라이다
  원점보다 앞에 표시된다.
- 자홍 십자: 차량 뒤에 장착된 라이다 원점

기본 디버그 화면은 차량 전진 방향이 위를 향하도록 `0°`로 표시한다. 이
설정은 표시 전용이며 검출 좌표를 바꾸지 않는다. 표시 방향만 바꾸려면
`--lidar-display-rotation`, 실제 센서 각도 보정은 `--lidar-angle-offset`을
사용한다.

LiDAR 장착 위치는 기본적으로 뒤 범퍼보다 10cm 뒤, 뒤 차축보다 30cm 뒤로
가정한다. 측정 후 JSON을 수정하거나 재생 명령에서
`--lidar-behind-vehicle-rear-cm 10 --lidar-to-rear-axle-cm -30`으로 바로
덮어쓸 수 있다. 뒤쪽 LiDAR 기준에서 차량 앞 방향은 음수이다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --lidar-display-rotation 0 `
  --lidar-angle-offset -90
```

주차칸 내부가 비어 있다는 대회 조건 때문에 내부 점군의 유무를 주차칸
판정에 사용하지 않는다. 빨강 ROI는 예상 밖 물체가 차량 바로 뒤에 들어온
경우만 정지시키는 별도 안전장치다.

## 상태 순서

```text
IDLE
  -> SEARCH_CARS
  -> TRACK_GAP (첫 차량을 추적하며 좌조향 시작 위치로 접근)
  -> PREALIGN_LEFT
  -> VERIFY_SLOT_BOX (다음 고유 LiDAR scan 1회로 ARM)
  -> SET_REVERSE_STEER (곡선 모드: 선택된 우조향을 정지 상태에서 설정)
  -> FOLLOW_ENTRY_CURVE (곡선 모드: 선택된 우조향으로 후진)
  -> RELEASE_ENTRY_STEER (정지 상태에서 조향 완화)
  -> FOLLOW_SLOT_CENTER
  -> PARK_CONFIRM (footprint 완료 후보 scan에서 즉시 정지)
  -> REACQUIRE_SLOT (최대 0.6초 동안 slot pose 복구)
  -> FINISH_REVERSE_TIMED (후진 시작 후 복구 실패 시 저속 마무리)
  -> PARKED
  -> EXIT_RIGHT
  -> EXIT_DONE
```

정상 주차 완료는 고정된 `950 x 1500 mm` slot-local frame에서 실측 차량
폭·길이와 LiDAR-뒤범퍼 오프셋으로 차체 네 모서리를 계산해 판단한다. 다만 뒤쪽
LiDAR가 주차칸 안에서 두 차량을 잃어 완료 판정 없이 멈추는 상황을 방지하기 위해,
이미 후진을 시작한 뒤 pose 복구가 0.6초 안에 되지 않으면 ARM된 진입 조향을
최대 0.3초만 유지하고 직선 저속 후진을 포함한 총 1.5초의
`FINISH_REVERSE_TIMED`로 마무리한다. 정렬 후 직선 후진 2.5초가 끝났거나
주차칸 뒤 여유에 도달한 경우에도 재탐색 대기 없이 `PARKED`로 확정한다.

대회 조건상 첫 번째 주차 차량 바로 옆이 빈 주차칸으로 보장되므로 두 번째
차량 검출을 기다리지 않는다. 첫 차량의 주차칸 인접 모서리를 2개 scan에서
확인하고 그 모서리가 기본 `yBack=-65cm`에 도달하면 최대 좌조향을 시작한다.
두 번째 차량은 좌조향 중에 주차칸 폭과 방향을 확정하는 데 사용한다.
첫 차량은 확장 ROI에서 최근접 군집으로 계속 추적한다. 첫 차량이 우측 후방
게이트를 지난 뒤에만 두 번째 차량 게이트를 ARM하므로, 차량 왼쪽에 있는
사람이나 다른 군집은 두 번째 차량으로 사용할 수 없다.
`POSITION_REAR_AXLE`은 선제 좌조향 기능을 끈 경우에만 사용하는 fallback이며,
그 경우에도 오차 부호에 따라 한 방향으로 보정하고 목표에 들어오면 즉시
다음 상태로 넘어가므로 앞뒤 왕복을 반복하지 않는다.

`PREALIGN_LEFT`에서는 먼저 정지한 채 설정된 최대 좌조향까지 0.4초 동안
조향한 뒤 저속으로 전진한다. 단순 타이머 회전이 아니라 매 LiDAR scan의
주차칸 깊이 방향, 뒤축에서 입구 중심까지의 방향과 거리를 확인한다. 충돌
없는 경로 후보가 한 번 나오면 즉시 정지하고 `VERIFY_SLOT_BOX`로 넘어간다.
여기서는 중복 loop를 세지 않고 다음 고유 LiDAR scan 한 번에서 경로를 ARM한다.
각도는 직접 후진과 곡선 후진 중 어느 모드를 사용할지만 선택하며 ARM 자체를
막지 않는다. 단, 전체 차체가 주차칸 입구부터 최종 정지 위치까지 통과할 수
있는 경로만 ARM한다. `VERIFY_SLOT_BOX`에서 이 검사가 실패하면 정지 상태로
끝내거나 `REACQUIRE_SLOT`을 반복하지 않고, 한 제어 주기 정지 후
`PREALIGN_LEFT`/`PROVISIONAL_PREALIGN` 전진을 재개한다.
곡선 후진은 `SET_REVERSE_STEER`에 진입했다는 사실만으로 시작하지 않는다.
`VERIFY_SLOT_BOX`에서 생성한 READY 경로와 선택 조향이 함께 ARM된 경우에만
후진 명령을 낸다. 조향 설정 전 또는 설정 중 slot pose가 소실되면 기존 ARM을
폐기하고, 정지 상태에서 pose를 다시 찾은 뒤 `VERIFY_SLOT_BOX`의 새 고유
scan으로 경로를 다시 ARM한다.

곡선 모드는 설정된 150/135/120/105/90 조향 후보에 대응하는 곡률을 모두
검사하고 가장 중앙에 안전하게 끝나는 후보를 고른다. 선택한 조향은 0.4초
정착 중 새 scan의 작은 흔들림으로 바꾸지 않는다. 최소 곡선 주행 시간이 지난
뒤 주차칸과의 각도 오차가 12도 이내인 고유 scan 한 번에서 조향 완화를
시작한다. 조향 완화 중에는 차를 세워 차체가 검사하지 않은 곡선을 그리지
않게 하고, 바퀴가 풀린 뒤 주차칸 중심 방향 후진을 재개한다.

사전정렬 중에는 아직 map을 잠그지 않고 매 scan에서 확인된 두 차량으로 동적
박스를 만든다. 안전 경로 후보가 나온 뒤 `VERIFY_SLOT_BOX`에서 정지한 다음
고유 scan의 `DIRECT_PAIR` pose로 처음 map을 freeze한다. 이후 raw pair가 다시
관측되어도 map의 source scan과 local 좌표는 바뀌지 않는다. map 정합이 정상일
때 direct-pair 차이가 이동 80mm·회전 5도 이내이면 차이의 25%만 보정하고, 큰
점프는 거부한다. 반대로 map 정합 자체가 실패한 경우에는 멈추지 않고 scan당
최대 80mm·5도만 direct-pair 방향으로 따라가는 `DIRECT_PAIR_RECOVERY`를
사용한다. 검증에 실패해 사전정렬로 돌아가면 승인되지 않은 map은 버리고 다음
검증에서 현재 두 차량으로 새 map을 만든다.

주차 시작 225초에는 소프트 데드라인을 적용한다. 이미 경로가 ARM됐거나 후진을
시작했다면 시간 기반 마무리로 전환하며, 기본 1.5초 마무리·3초 주차 유지·1.6초
출차 회전을 포함해 240초 전에 `EXIT_DONE`에 도달하도록 예산을 잡는다. LiDAR
통신 단절 또는 실제 안전 ROI 장애물은 이 완주용 fallback보다 우선하며 기존처럼
즉시 오류 정지한다.

녹화 영상은 이미 정해진 차량 움직임을 바꿀 수 없으므로 명령의 상태 전이와
가상 속도/조향만 검증할 수 있다. 실제 차량에서는 아래 값을 바퀴를 띄운
상태와 넓은 빈 공간에서 먼저 보정한다.

- `--prealign-speed`: 선제·일반 좌회전 전진 구간을 함께 덮어쓴다. 현재 JSON의
  선제 사전정렬 속도는 35이다.
- `--prealign-steering`: 선회 준비 전용 좌회전 명령. 기본값 -120. 후진 경로
  추종 상한 `max_steering=150`과는 독립적이다.
- `--prealign-timeout-s`: 정렬 실패 후 LiDAR 곡선 경로 전환을 검토할 시간. 기본값 6초
- `--first-car-turn-target-cm`: 첫 차량 모서리의 좌조향 시작 좌표. 기본값 -65cm

`scripts/arduino_parking_replay.py`는 더 이상 별도의 Python 상태 머신으로
속도와 조향을 재계산하지 않는다. 실시간 `scripts/parking.py`와 동일한
`TParkingPlanner`를 직접 호출하며, CSV의 `planner_state`, `drive_speed`,
`steer_deg`, `event`는 실시간 실행에서 생성될 값과 동일하다. 단, 녹화된
차량 궤적은 가상 명령에 반응하지 않으므로 상태가 timeout으로 끝날 수 있다.

주차 런타임은 초음파 스트림을 켜거나 측정값을 읽지 않으며, 탐지·조향·정지
판단에는 라이다만 사용한다. 최초 정상 라이다 scan 전에는 정지 상태로
기다린다. 정상 scan을 한 번 받은 뒤 라이다가 끊기거나 stale/error 상태가
되면 즉시 `EMERGENCY_STOP` 명령을 보내고 런타임을 오류 코드로 종료한다.

후진 중 빨강 안전 ROI에 물체가 들어오면 `EMERGENCY_STOP`이 걸리고 운전자
reset 전까지 해제되지 않는다.

## 후진 경로

BEV에서 현재 후축 위치, 가상 뒷선 앞 정지 목표, 라이다 주차칸 방향을 이용해
`일정 곡률 원호 + 주차칸 방향 직선`의 전체 경로를 만든다. 각 조향 후보마다
실측 `600 × 1000mm` 차체 외곽을 8px 이하 간격으로 경로 전체에 배치해 양옆
차량 경계와 가상 뒷선을 넘는지 검사한다. 최종 차체 앞쪽이 주차칸 안으로
20px 이상 들어오고, 측면 최소 여유가 8px 이상인 후보만 `READY`다. 따라서
차체가 아직 입구 앞에만 있는 짧은 원호는 더 이상 `READY`가 될 수 없다.

`entry_steering_ratio_candidates`는 기본
`[1.0, 0.9, 0.8, 0.7, 0.6]`이며 `max_steering=150`일 때
150/135/120/105/90 명령에 해당한다. 실제 명령과 회전 반경의 관계는
`maximum_curvature_per_px`와 `full_steering_curvature_per_px`로 보정한다.
다음 실험에서는 조향 후보 목록보다 이 곡률 보정이 먼저 맞는지 확인해야 한다.

세션 CSV에는 선택 비율, 예상 회전 각도, 최종 횡오차, 최소 측면 여유,
최대 진입 깊이가 각각 `path_entry_steering_ratio`,
`path_entry_heading_change_deg`, `path_final_lateral_offset_px`,
`path_minimum_side_clearance_px`, `path_maximum_entry_depth_px`로 저장된다.

## 현재 녹화 라이다 검증 결과

제공된 첫 로그는 456 scan, 52.44초, 약 8.7 Hz이다. 보정 전 화면에서는
오른쪽 축이 실제 차량 앞, 위쪽 축이 실제 차량 왼쪽으로 나타났다. raw 90°를
차량 앞 0°로 맞추기 위해 `angle_offset_deg=-90`,
`clockwise_angles=false`를 사용한다. 디버그 화면은 차량 앞=위, 오른쪽=오른쪽,
뒤=아래, 왼쪽=왼쪽으로 고정한다. 현재 보정값에서:

- 차량 오른쪽의 두 군집 간격이 처음 연속 확인되는 시점: 약 22.5초
- 최초 확인 간격: 약 1523 mm
- 최초 확인 시 임시 후축 보정값 기준 오차: 약 +26 mm

이 수치는 녹화 로그의 상대 검증값이다. 실제 차량 구동 전 아래 차량 치수를
반드시 측정해 `configs/parking.json`에 반영해야 한다.

1. 라이다 원점에서 후축 중심까지의 부호 있는 전후 거리(현재 임시값 `-300mm`)
2. 후방카메라에서 후축 중심까지의 전후 거리, 카메라 높이와 아래보기 각도
3. 차량 전체 폭, 축거, 후방 오버행
4. 최대 조향각과 조향 명령 부호

## BEV 조정

현재 시작값은 원거리 과확대를 줄이기 위해 far edge를 낮고 넓게 잡았다.

```powershell
python scripts/bev_tune.py `
  --source 녹화파일.mp4 `
  --no-centerline `
  --out-width 600 `
  --out-height 600 `
  --dst-margin 0.15 `
  --top-y 0.56 `
  --top-left-x 0.18 `
  --top-right-x 0.82 `
  --bottom-y 1.0 `
  --bottom-left-x 0.0 `
  --bottom-right-x 1.0
```

`p`를 눌러 출력된 비율을 `configs/parking.json`에 복사한다. 실제 평행한
주차선이 BEV에서도 평행하고, 주차칸 폭이 화면 상단과 하단에서 거의 같아야
한다.

`parking.py`에서도 설정 파일을 수정하지 않고 같은 항목을 덮어쓸 수 있다.
제공 영상에서 먼 주차선을 더 포함시키기 위한 시작 예시는 다음과 같다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --recording-zip "첫번째 라이다 데이터.zip" `
  --device cpu `
  --imgsz 512 `
  --bev-top-y 0.25 `
  --bev-top-left-x 0.00 `
  --bev-top-right-x 1.00 `
  --bev-bottom-y 1.00 `
  --bev-bottom-left-x 0.00 `
  --bev-bottom-right-x 1.00 `
  --bev-dst-margin 0.15
```

- `--bev-top-y`를 낮추면 더 먼 영역이 보이지만 상단 확대가 강해질 수 있다.
- top-left를 낮추고 top-right를 높여 상단 폭을 넓히면 상단 확대가 줄어든다.
- `--bev-dst-margin`을 줄이면 BEV 좌우 검은 여백이 줄어든다.
- `--bev-out-width`, `--bev-out-height`로 출력 화면 크기도 바꿀 수 있다.

x 좌표는 `-1.0~2.0`까지 입력할 수 있다. `left < 0`, `right > 1`은 영상보다
넓은 영역을 source로 잡기 때문에 좌우 시야는 넓어지지만 영상 밖 검은 영역이
생길 수 있다. 반대로 BEV 속 물체 자체를 가로로 더 크게 보이게 하려면
`left > 0`, `right < 1`처럼 source 폭을 좁혀야 한다. `bev_tune.py`의 x
트랙바도 동일하게 `-1.0~2.0` 범위를 사용한다.

## 실차 실행

Apple Silicon에서는 `--device auto`가 MPS를 선택한다. 모터 출력은 녹화
검증과 바퀴를 띄운 방향 확인 후에만 명시적으로 켠다.

```powershell
..\venv\Scripts\python.exe scripts/parking.py `
  --source 1 `
  --lidar-port COM5 `
  --model trained_model/parking_best.pt `
  --device mps `
  --serial `
  --serial-port COM6
```

### 실차 분석 세션 저장

라이브 실행은 기본값 `--record-session auto`에 의해 다음 파일을
`data/parking_sessions/<실행시각>/`에 함께 저장한다.

- `*_dashboard.mp4`: 실제 표시된 카메라·BEV·LiDAR·상태 화면
- `*_lidar.csv`: 원본 LiDAR 스캔(`timestamp, quality, angle, distance`)
- `*_telemetry.csv`: 상태, 계획/실제 명령, 슬롯 pose, path 판정,
  motion lease, 주차 완료 판정
- `*_metadata.json`: 적용 설정, 실행 옵션, 종료 상태와 데이터 개수
- `*_replay.zip`: 위 MP4와 LiDAR CSV를 바로 replay할 수 있는 묶음

초음파는 판단에도 저장에도 사용하지 않는다. 정상 종료뿐 아니라 `Q`,
`Esc`, 창 닫기 또는 LiDAR 오류로 종료할 때도 지금까지 받은 CSV와
메타데이터를 닫아 보존한다. replay ZIP은 대시보드 영상과 LiDAR 스캔이
둘 다 존재할 때 생성된다. 아래 명령의 `--record-camera`는 카메라 영상을
대시보드에 함께 저장하기만 한다. `--no-camera`이므로 YOLO는 로드하지
않고 주차 판단과 차량 명령은 계속 LiDAR만 사용한다.

Windows에서 포트를 확인한다.

```powershell
python scripts/list_serial_ports.py
```

먼저 모터 출력을 끈 상태로 LiDAR와 기록을 확인한다.

```powershell
python scripts/parking.py `
  --source 1 `
  --no-camera `
  --record-camera `
  --lidar-port COM5 `
  --record-session on `
  --record-dashboard on
```

그다음 바퀴 방향과 정지 동작을 확인한 뒤 실차 주차를 실행한다.

```powershell
python scripts/parking.py `
  --source 1 `
  --no-camera `
  --record-camera `
  --lidar-port COM5 `
  --serial `
  --serial-port COM6 `
  --record-session on `
  --record-dashboard on
```

화면이 열린 뒤 `Space`로 미션을 시작한다. 이상이 있으면 `R`로 즉시
정지·초기화하고, `Q` 또는 `Esc`로 종료한다. COM 번호는 당일
`list_serial_ports.py` 출력에 맞게 바꾼다.

macOS 실차 실행 예시는 다음과 같다. `--prealign-steering -120`은 현재
`vehicle_controller.ino`의 좌측 끝 pot 보정값으로 이동하므로, 먼저 바퀴를
띄운 상태에서 실제 기구 끝과 일치하는지 확인한다.

```bash
python3 scripts/parking.py \
  --source 1 \
  --lidar-port /dev/cu.usbserial-LIDAR \
  --model trained_model/parking_best.pt \
  --device mps \
  --serial \
  --serial-port /dev/cu.usbmodem-ARDUINO \
  --prealign-speed 35 \
  --prealign-steering -120 \
  --prealign-timeout-s 6
```
