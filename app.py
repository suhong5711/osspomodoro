import streamlit as st
import streamlit.components.v1 as components
import time
import cv2
import pathlib
import numpy as np
import sys
from datetime import datetime
from ultralytics import YOLO

if sys.platform == 'win32':
    pathlib.PosixPath = pathlib.WindowsPath

MODEL_PATH = 'C:/Users/LG/OneDrive/Documents/GitHub/software-os/opensw/opensw/openswbest3.pt'
model = YOLO(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6

TIMER_CSS_RED = """<style>
.circle {
  width: 100px;
  height: 100px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}
.circle span {
  font: 600 1.2rem monospace;
  color: #fff;
}
</style>"""

TIMER_CSS_BLUE = TIMER_CSS_RED

def draw_circle(remaining, total, color="red"):
    pct = remaining / total if total else 0
    angle = pct * 360
    mm, ss = divmod(remaining, 60)
    css = TIMER_CSS_RED if color == "red" else TIMER_CSS_BLUE
    grad_color = "#e74c3c" if color == "red" else "#3498db"
    html = css + f"""
    <div class='circle'
         style='background:
            conic-gradient({grad_color} 0deg {angle}deg,
                           #eeeeee {angle}deg 360deg);'>
      <span>{mm:02d}:{ss:02d}</span>
    </div>"""
    return html

def init_state():
    defaults = {
        "cap": None,
        "running": False,
        "paused": False,
        "time_left": 0,
        "set_index": 1,
        "cycle_type": "focus",
        "start_requested": False,
        "stop_flag": False,
        "start_time": None,
        "hand_time": 0,
        "phone_time": 0,
        "neutral_time": 0,
        "completed": False,
        "last_frame_time": None,
        "startup_latency": 0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.sidebar.title("설정")
focus_sec = st.sidebar.number_input("지비중 시간 (초)", 10, 3600, 20)
break_sec = st.sidebar.number_input("쉬는 시간 (초)", 1, 1800, 5)
total_sets = st.sidebar.number_input("세트 수", 1, 10, 2)
st.sidebar.text_area("📜 오늘 할 일 목록")

btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
with btn1:
    if st.button("▶ 시작"):
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = focus_sec
        st.session_state.start_requested = True
        st.session_state.stop_flag = False
        st.session_state.completed = False
        st.session_state.start_time = time.time()
        st.session_state.hand_time = 0
        st.session_state.phone_time = 0
        st.session_state.neutral_time = 0
        st.session_state.last_frame_time = time.time()
        cap_start = time.time()
        st.session_state.cap = cv2.VideoCapture(0)
        while not st.session_state.cap.isOpened():
            pass
        st.session_state.startup_latency = time.time() - cap_start

with btn2:
    if st.button("⏯ 정지/재시작"):
        if st.session_state.running:
            st.session_state.running = False
            st.session_state.paused = True
        elif st.session_state.paused and st.session_state.time_left > 0:
            st.session_state.running = True
            st.session_state.paused = False

with btn3:
    if st.button("🔄 초기화"):
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = focus_sec
        st.session_state.stop_flag = False
        st.session_state.completed = False

with btn4:
    if st.button("⏹ 중지/재시작"):
        if not st.session_state.stop_flag:
            st.session_state.running = False
            st.session_state.paused = False
            if st.session_state.cycle_type == "focus":
                st.session_state.time_left = focus_sec
            else:
                st.session_state.time_left = break_sec
            st.session_state.stop_flag = True
        else:
            st.session_state.running = True
            st.session_state.paused = False
            st.session_state.stop_flag = False

colL, colR = st.columns([1, 3])
status_placeholder = st.empty()
with colL:
    st.markdown("### 🔵 타이머")
    container_timer = st.empty()
with colR:
    container_video = st.empty()
    status_text = st.empty()

def show_frame():
    ret, frame = st.session_state.cap.read()
    if not ret:
        return

    now = time.time()
    delta = now - st.session_state.last_frame_time
    st.session_state.last_frame_time = now

    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
    found = False
    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        color = (255, 0, 0) if cls_name == 'hand_with_pen' else (0, 0, 255) if cls_name == 'smartphone' else (255, 255, 255)
        if cls_name == 'hand_with_pen':
            st.session_state.hand_time += delta
        elif cls_name == 'smartphone':
            st.session_state.phone_time += delta
        found = True
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if not found:
        st.session_state.neutral_time += delta
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    container_video.image(rgb, channels="RGB")

def update_timer_ui(duration):
    color = "blue" if st.session_state.cycle_type == "break" else "red"
    with container_timer:
        components.html(draw_circle(st.session_state.time_left, duration, color), height=120)

def run_timer(duration):
    end_time = time.time() + st.session_state.time_left
    while time.time() < end_time:
        if not st.session_state.running:
            update_timer_ui(duration)
            time.sleep(0.05)
            continue
        st.session_state.time_left = int(end_time - time.time())
        update_timer_ui(duration)
        show_frame()
        status = f"{total_sets}세트 중 {st.session_state.set_index}세트 {('지비중' if st.session_state.cycle_type == 'focus' else '휴식중')}"
        status_text.subheader(status)
        loop_time = 0.1
        time.sleep(loop_time)

def run_timer_cycle():
    while st.session_state.running and st.session_state.set_index <= total_sets:
        duration = focus_sec if st.session_state.cycle_type == "focus" else break_sec
        run_timer(duration)

        if not st.session_state.running or st.session_state.paused:
            break

        if st.session_state.cycle_type == "focus":
            st.session_state.cycle_type = "break"
            st.session_state.time_left = break_sec
        else:
            st.session_state.set_index += 1
            if st.session_state.set_index > total_sets:
                st.session_state.running = False
                st.session_state.completed = True
                status_placeholder.success("🎉 모든 세트 완료!")
                break
            else:
                st.session_state.cycle_type = "focus"
                st.session_state.time_left = focus_sec

if st.session_state.running:
    if st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
    run_timer_cycle()
else:
    update_timer_ui(focus_sec if st.session_state.cycle_type == "focus" else break_sec)
    if st.session_state.running and not st.session_state.paused:
        show_frame()
    elif st.session_state.paused:
        container_video.markdown("⏸ 지금은 정지 상태입니다. 시작하려면 재시작을 누르세요.")
    elif st.session_state.stop_flag:
        container_video.markdown("⏹ 지금은 중지 상태입니다. 세트를 시작하려면 중지/재시작을 누르세요.")

if st.session_state.completed:
    detect_total = st.session_state.hand_time + st.session_state.phone_time + st.session_state.neutral_time
    adjusted_detect_total = detect_total - st.session_state.startup_latency
    expected_total = total_sets * (focus_sec + break_sec)
    adjusted_neutral = st.session_state.neutral_time - st.session_state.startup_latency
    err_time = expected_total - adjusted_detect_total
    m, s = divmod(int(expected_total), 60)
    st.markdown("### 🏁 결과 요약")
    st.table({
        "항목": ["총 타이머 시간", "펜 인식 시간", "휴대폰 인식 시간", "미탐지 시간 (보정)", "세트 수", "오차 시간", "탐지 시간 합계 (보정)"],
        "값": [f"{m:02d}:{s:02d}",
              f"{st.session_state.hand_time:.1f}s",
              f"{st.session_state.phone_time:.1f}s",
              f"{adjusted_neutral:.1f}s",
              total_sets,
              f"{err_time:.1f}s",
              f"{adjusted_detect_total:.1f}s"]
    })
