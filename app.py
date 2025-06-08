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

MODEL_PATH = r'C:\Users\user\Desktop\opensw\opensw\openswbest3.pt'
model = YOLO(MODEL_PATH)
CONFIDENCE_THRESHOLD = 0.4
IOU_THRESHOLD = 0.6

TIMER_CSS = """<style>
.circle {
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

def draw_circle(remaining, total):
    pct = remaining / total if total else 0
    angle = pct * 360
    mm, ss = divmod(remaining, 60)
    html = TIMER_CSS + f"""
    <div class='circle'
         style='background:
            conic-gradient(#e74c3c 0deg {angle}deg,
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
        "start_requested": False
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

st.sidebar.title("설정")
focus_min = st.sidebar.number_input("집중 시간 (분)", 1, 60, 25)
break_min = st.sidebar.number_input("쉬는 시간 (분)", 1, 30, 5)
total_sets = st.sidebar.number_input("세트 수", 1, 10, 3)
focus_sec = focus_min * 60
break_sec = break_min * 60

btn1, btn2, btn3, btn4 = st.columns([1, 1, 1, 1])
with btn1:
    if st.button("▶ 시작"):
        st.session_state.running = True
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = focus_sec
        st.session_state.start_requested = True
with btn2:
    if st.button("⏸ 정지"):
        st.session_state.running = False
        st.session_state.paused = True
with btn3:
    if st.button("🔄 초기화"):
        st.session_state.running = False
        st.session_state.paused = False
        st.session_state.set_index = 1
        st.session_state.cycle_type = "focus"
        st.session_state.time_left = 0
with btn4:
    if st.button("🔁 재시작"):
        if st.session_state.paused and st.session_state.time_left > 0:
            st.session_state.running = True
            st.session_state.paused = False

colL, colR = st.columns([1, 3])
status_placeholder = st.empty()
with colL:
    st.markdown("### 🔵 타이머")
    container_timer = st.empty()
with colR:
    container_video = st.empty()
    status_text = st.empty()

if st.session_state.cap is None:
    st.session_state.cap = cv2.VideoCapture(0)

def show_frame():
    ret, frame = st.session_state.cap.read()
    if not ret:
        return
    results = model(frame, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
    for box in results.boxes:
        cls_id = int(box.cls.item())
        cls_name = model.names[cls_id]
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        color = (255, 0, 0) if cls_name == 'hand_with_pen' else (0, 0, 255) if cls_name == 'smartphone' else (255, 255, 255)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, cls_name, (xmin, max(0, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    container_video.image(rgb, channels="RGB")

def update_timer_ui(duration):
    with container_timer:
        components.html(draw_circle(st.session_state.time_left, duration), height=120)

def run_timer(duration):
    end_time = time.time() + st.session_state.time_left
    while time.time() < end_time:
        if not st.session_state.running:
            update_timer_ui(duration)
            time.sleep(0.1)
            continue
        st.session_state.time_left = int(end_time - time.time())
        update_timer_ui(duration)
        show_frame()
        status_text.subheader(f"{total_sets}세트 중 {st.session_state.set_index}세트 진행중")
        time.sleep(0.1)

if st.session_state.running:
    duration = focus_sec if st.session_state.cycle_type == "focus" else break_sec
    run_timer(duration)
    if not st.session_state.paused:
        if st.session_state.cycle_type == "focus":
            st.session_state.cycle_type = "break"
            st.session_state.time_left = break_sec
        else:
            st.session_state.set_index += 1
            if st.session_state.set_index > total_sets:
                st.session_state.running = False
                status_placeholder.success("🎉 모든 세트 완료!")
            else:
                st.session_state.cycle_type = "focus"
                st.session_state.time_left = focus_sec
else:
    update_timer_ui(focus_sec if st.session_state.cycle_type == "focus" else break_sec)
    if st.session_state.paused:
        container_video.markdown("⏸ 정지 상태입니다. 재시작을 누르면 다시 시작됩니다.")
    else:
        show_frame()
