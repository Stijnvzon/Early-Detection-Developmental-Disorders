# reference_builder.py

import os
import json
import uuid
import tempfile
from datetime import datetime

import cv2
import streamlit as st

# -----------------------------
# Configuration
# -----------------------------
REFERENCE_DIR = "reference_frames"
METADATA_PATH = os.path.join(REFERENCE_DIR, "metadata.json")
os.makedirs(REFERENCE_DIR, exist_ok=True)


# -----------------------------
# Helper functions
# -----------------------------
def save_reference_frame(frame_bgr, description, source_video, frame_index):
    """
    Save a selected frame and its description to REFERENCE_DIR
    and append a record to metadata.json.
    """
    # Create a unique filename
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rnd = uuid.uuid4().hex[:6]
    filename = f"ref_{ts}_{rnd}.jpg"
    save_path = os.path.join(REFERENCE_DIR, filename)

    # Save the BGR frame to disk
    cv2.imwrite(save_path, frame_bgr)

    # Load existing metadata if it exists
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    # Append new entry
    metadata.append(
        {
            "path": filename,          # relative path from REFERENCE_DIR
            "description": description,
            "source_video": source_video,
            "frame_index": int(frame_index),
            "created_at": ts,
        }
    )

    # Save updated metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_metadata():
    """Load all stored reference frame metadata."""
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Reference Builder – Select Example Frames")

st.write(
    "Upload an **example video** (for instance from a confirmed or clear case), "
    "choose frames using the slider, label them with Relevance / Support / Assistance, "
    "and add a short Observation. The frame + description will be saved as reference "
    "examples for your main analysis app."
)

uploaded_video = st.file_uploader(
    "Upload example video",
    type=["mp4", "mov", "avi"],
    key="example_video",
)

if uploaded_video is not None:
    # Keep video bytes in session_state so the slider works smoothly
    if (
        "video_bytes" not in st.session_state
        or st.session_state.get("video_name") != uploaded_video.name
    ):
        st.session_state["video_bytes"] = uploaded_video.read()
        st.session_state["video_name"] = uploaded_video.name

    # Write bytes to a temporary file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(st.session_state["video_bytes"])
        tmp_video_path = tmp_file.name

    # Open video
    cap = cv2.VideoCapture(tmp_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    if total_frames == 0:
        st.error("Could not read any frames from this video.")
        cap.release()
    else:
        st.write(f"Total frames in this video: **{total_frames}**")

        # Slider step (for long videos, we don't need every single frame)
        step = max(total_frames // 200, 1)

        frame_idx = st.slider(
            "Choose a frame index",
            min_value=0,
            max_value=total_frames - 1,
            value=total_frames // 2,
            step=step,
        )

        # Seek to the chosen frame and read it
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        cap.release()
        os.remove(tmp_video_path)  # clean up temp video file

        if not ret:
            st.error("Could not read this frame from the video.")
        else:
            # Convert BGR -> RGB for display
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            st.image(
                frame_rgb,
                caption=f"Selected frame index: {frame_idx}",
                use_container_width=True,
            )

            st.markdown(
                "We will build the full description automatically with this structure:\n\n"
                "`Relevance: <value>. Support: <value>. Assistance: <value>. Observation: <your sentence>.`"
            )

            # Label controls
            col0, col1, col2 = st.columns(3)

            with col0:
                relevance_label = st.radio(
                    "Relevance (is this frame useful for analysis?)",
                    options=["relevant", "skip"],
                    index=0,
                    help="Use 'skip' for frames that are not informative (e.g. child off-screen, no movement, wrong angle).",
                )

            with col1:
                support_label = st.radio(
                    "Support (use of arms/hands to rise/keep balance?)",
                    options=["yes", "no", "uncertain"],
                    index=0,
                    help="Does the child clearly use arms/hands on floor, legs, furniture, or another person to rise or keep balance?",
                )

            with col2:
                assistance_label = st.radio(
                    "Assistance (external help or object needed?)",
                    options=["yes", "no", "uncertain"],
                    index=0,
                    help="Is the child clearly relying on external help (person, furniture, rail, etc.) to perform the movement?",
                )

            observation = st.text_area(
                "Observation (objective, ≤ 1–2 short sentences)",
                placeholder=(
                    "Example: Child pushes on knees with both hands while rising from the floor."
                ),
            )

            if st.button("Save this frame as reference"):
                if not observation.strip():
                    st.error("Please fill in the Observation before saving.")
                else:
                    full_description = (
                        f"Relevance: {relevance_label}. "
                        f"Support: {support_label}. "
                        f"Assistance: {assistance_label}. "
                        f"Observation: {observation.strip()}"
                    )

                    save_reference_frame(
                        frame_bgr,
                        full_description,
                        source_video=st.session_state.get("video_name"),
                        frame_index=frame_idx,
                    )
                    st.success("Reference frame saved in 'reference_frames/'.")

else:
    st.info("Upload an example video above to start creating reference frames.")

# -----------------------------
# Overview of stored reference frames
# -----------------------------
st.header("Stored reference frames")

metadata = load_metadata()
if not metadata:
    st.write("No reference frames saved yet.")
else:
    for i, item in enumerate(metadata, start=1):
        img_path = os.path.join(REFERENCE_DIR, item["path"])
        st.markdown(f"**Reference {i}**")
        if os.path.exists(img_path):
            st.image(img_path, width=300)
        else:
            st.warning(f"Image file not found: {img_path}")

        st.write(f"- Description: {item.get('description', '')}")
        st.write(f"- Source video: {item.get('source_video', 'unknown')}")
        st.write(f"- Frame index: {item.get('frame_index', 'unknown')}")
        st.write(f"- Created at: {item.get('created_at', 'unknown')}")
        st.markdown("---")
