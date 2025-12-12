# app.py

import os
import json
import base64
import tempfile
import cv2
import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration & paths
# ---------------------------------------------------------

REFERENCE_DIR = "reference_frames"
METADATA_PATH = os.path.join(REFERENCE_DIR, "metadata.json")
os.makedirs(REFERENCE_DIR, exist_ok=True)

# Adjust these if your Ollama endpoint is different
LLAVA_URL = "http://localhost:11434/api/generate"
LLAMA3_URL = "http://localhost:11434/api/generate"


# ---------------------------------------------------------
# Reference example loading
# ---------------------------------------------------------

def load_reference_examples():
    """
    Load reference images and descriptions from metadata.json,
    and encode images to base64 strings for LLaVA.
    Returns (images_b64_list, descriptions_list).
    """
    if not os.path.exists(METADATA_PATH):
        return [], []

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    images_b64 = []
    descriptions = []

    for item in metadata:
        img_rel = item.get("path")
        desc = item.get("description", "")
        if not img_rel:
            continue

        # Resolve path
        img_path = img_rel
        if not os.path.isabs(img_rel):
            img_path = os.path.join(REFERENCE_DIR, img_rel)

        if not os.path.exists(img_path):
            st.warning(f"Reference image not found: {img_path}")
            continue

        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        images_b64.append(img_b64)
        descriptions.append(desc)

    return images_b64, descriptions


def load_reference_metadata():
    """Load raw metadata for UI display."""
    if not os.path.exists(METADATA_PATH):
        return []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------
# Motion-based frame selection
# ---------------------------------------------------------

def select_informative_frame_indices(
    video_path,
    max_frames=8,
    sample_stride=2,
    min_gap_seconds=1.0,
):
    """
    Select 'informative' frame indices based on motion between consecutive frames.

    - Compute a simple motion score = mean absolute pixel difference
      between consecutive grayscale frames (sampled every `sample_stride` frames).
    - Pick the top `max_frames` moments with the highest motion.
    - Enforce at least `min_gap_seconds` between chosen frames so they
      are spread across the video.

    Returns:
        A sorted list of frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0  # safe fallback

    min_gap_frames = int(min_gap_seconds * fps)

    scores = []  # list of (frame_idx, score)
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optional: skip some frames for speed
        if frame_idx % sample_stride != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            score = float(diff.mean())  # simple motion metric
            scores.append((frame_idx, score))

        prev_gray = gray
        frame_idx += 1

    cap.release()

    if not scores:
        return []

    # Sort by motion score (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Greedily pick top-scoring frames, enforcing a minimum gap
    selected = []
    for idx, score in scores:
        if len(selected) >= max_frames:
            break
        if all(abs(idx - s) >= min_gap_frames for s in selected):
            selected.append(idx)

    selected.sort()
    return selected


# ---------------------------------------------------------
# Prompt builder for LLaVA
# ---------------------------------------------------------

def build_llava_prompt(reference_descriptions):
    """
    Build the instruction prompt for LLaVA.
    The reference descriptions are used as 'Example 1, Example 2, ...'.
    """
    base_instruction = (
        "You are a clinical movement observer analyzing ONE target video frame of a child.\n"
        "Focus ONLY on gross motor movements and posture.\n"
        "Pay special attention to ANY use of arms/hands to rise, stand, or keep balance — "
        "including pushing or pulling on the floor, legs/knees, furniture, walls, playground equipment (e.g., a slide), "
        "rails, or another person's hand/arm. Do not speculate about diagnoses. Ignore identity, emotions, clothing, "
        "and irrelevant background details.\n\n"
    )

    if reference_descriptions:
        examples_text = "\n".join(
            f"- Example {i+1}: {desc}"
            for i, desc in enumerate(reference_descriptions)
        )
        numbered_images = (
            f"You will see {len(reference_descriptions) + 1} images in total:\n"
            f"- The first {len(reference_descriptions)} images are EXAMPLES to show what matters:\n"
            f"{examples_text}\n"
            "- The LAST image is the TARGET frame from the video that you must rate.\n\n"
        )
    else:
        numbered_images = (
            "You will see 1 image: it is the TARGET frame from the video that you must rate.\n\n"
        )

    output_format = (
        "Reply ONLY for the LAST (target) image, in EXACTLY four short lines (no extra text):\n"
        "Relevance: [relevant/skip]\n"
        "Support: [yes/no/uncertain]\n"
        "Assistance: [yes/no/uncertain]\n"
        "Observation: <= 20 words, objective\n"
    )

    return base_instruction + numbered_images + output_format


# ---------------------------------------------------------
# Model call helpers
# ---------------------------------------------------------

def call_llava(images_payload, prompt):
    """
    Call LLaVA with a list of images (base64 strings) and a text prompt.
    The last image in images_payload should be the target frame.
    """
    try:
        resp = requests.post(
            LLAVA_URL,
            json={
                "model": "llava", 
                "prompt": prompt,
                "images": images_payload,
                "stream": True,
            },
            stream=True,
            timeout=(30, 600),  # (connect_timeout, read_timeout)
        )
        resp.raise_for_status()
    except requests.exceptions.ReadTimeout as e:
        st.error(f"Error calling LLaVA (read timed out): {e}")
        return "Relevance: skip\nSupport: uncertain\nAssistance: uncertain\nObservation: LLaVA timed out on this frame."
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling LLaVA: {e}")
        return "Relevance: skip\nSupport: uncertain\nAssistance: uncertain\nObservation: LLaVA request failed on this frame."

    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            part = line.decode("utf-8")
            json_part = json.loads(part)
            text += json_part.get("response", "")
        except Exception:
            continue
    return text


def call_llama3(prompt):
    """Call Llama3 (LLM) with the given prompt and return the full text response."""
    try:
        resp = requests.post(
            LLAMA3_URL,
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Llama3: {e}")
        return ""

    text = ""
    for line in resp.iter_lines():
        if not line:
            continue
        try:
            part = line.decode("utf-8")
            json_part = json.loads(part)
            text += json_part.get("response", "")
        except Exception:
            continue
    return text


# ---------------------------------------------------------
# Streamlit UI – main analysis app
# ---------------------------------------------------------

st.title("Early Detection Helper – Video Analysis with Reference Examples")

st.write(
    "This tool uses example frames (with descriptions) created via `ref_frames.py` "
    "to provide more consistent visual context when analyzing new child videos.\n\n"
    "**Important:** This is **not** a diagnostic tool. It can only suggest whether it might be "
    "useful to seek medical evaluation. Always consult qualified healthcare professionals."
)

# 1) Load reference examples
ref_images_b64, ref_descriptions = load_reference_examples()
llava_prompt = build_llava_prompt(ref_descriptions)

if ref_images_b64:
    st.success(f"{len(ref_images_b64)} reference frames loaded and used as visual examples.")
else:
    st.warning(
        "No reference frames found. The model will rely only on the text prompt. "
        "Consider creating reference frames first with `reference_builder.py`."
    )

# Optional: display reference examples
with st.expander("Show reference examples used by the model"):
    metadata = load_reference_metadata()
    if not metadata:
        st.write("No reference frames stored yet.")
    else:
        for i, item in enumerate(metadata, start=1):
            img_path = os.path.join(REFERENCE_DIR, item["path"])
            st.markdown(f"**Reference {i}**")
            if os.path.exists(img_path):
                st.image(img_path, width=300)
            else:
                st.warning(f"Image missing: {img_path}")
            st.write(f"- Description: {item.get('description', '')}")
            st.write(f"- Source video: {item.get('source_video', 'unknown')}")
            st.write(f"- Frame index: {item.get('frame_index', 'unknown')}")
            st.write(f"- Created at: {item.get('created_at', 'unknown')}")
            st.markdown("---")

# 2) Upload and analyze a new video
st.header("Analyze a new child video")

uploaded_video = st.file_uploader(
    "Upload a video of the child",
    type=["mp4", "mov", "avi"],
    key="child_video",
)

if uploaded_video is not None and st.button("Start analysis"):
    st.info("Processing video...")

    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    try:
        with tempfile.TemporaryDirectory() as frames_dir:
            # ---------------------------------------------
            # 1) Select informative frames based on motion
            # ---------------------------------------------
            selected_indices = select_informative_frame_indices(
                video_path=video_path,
                max_frames=8,        # hard cap on frames sent to LLaVA
                sample_stride=2,     # score every 2nd frame
                min_gap_seconds=1.0, # at least 1 second apart
            )

            saved_frames = []

            if not selected_indices:
                # Fallback to simple interval sampling if motion-based selection fails
                st.warning(
                    "Motion-based selection found no frames; falling back to simple interval sampling."
                )
                cap = cv2.VideoCapture(video_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
                frame_interval = 2 * fps

                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % frame_interval == 0:
                        frame_filename = os.path.join(frames_dir, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        saved_frames.append(frame_filename)
                    frame_count += 1

                cap.release()
            else:
                st.write(f"Selected {len(selected_indices)} informative frames based on motion.")
                st.caption(f"Frame indices: {selected_indices}")

                # Read only those selected frames and save them as images
                cap = cv2.VideoCapture(video_path)
                selected_set = set(selected_indices)
                frame_idx = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in selected_set:
                        frame_filename = os.path.join(frames_dir, f"frame_{frame_idx}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        saved_frames.append(frame_filename)
                    frame_idx += 1

                cap.release()

            # ---------------------------------------------
            # 2) If no frames, stop; otherwise call LLaVA
            # ---------------------------------------------
            if not saved_frames:
                st.error("No frames extracted from video.")
            else:
                st.write(f"Extracted {len(saved_frames)} frames for analysis.")

                # Show thumbnails of the selected frames
                with st.expander("Show selected frames (thumbnails)"):
                    cols = st.columns(3)  # 3 images per row
                    for i, frame_path in enumerate(saved_frames):
                        col = cols[i % 3]

                        # Extract frame index from filename: "frame_<idx>.jpg"
                        try:
                            base = os.path.basename(frame_path)
                            idx_str = base.replace("frame_", "").split(".")[0]
                            frame_idx = int(idx_str)
                            caption = f"Frame index {frame_idx}"
                        except Exception:
                            caption = os.path.basename(frame_path)

                        col.image(frame_path, caption=caption, use_container_width=True)
                # Call LLaVA for each extracted frame
                descriptions = []
                progress = st.progress(0)
                total = len(saved_frames)

                for idx, frame_file in enumerate(saved_frames, start=1):
                    with open(frame_file, "rb") as f:
                        image_bytes = f.read()
                    target_b64 = base64.b64encode(image_bytes).decode("utf-8")

                    # For LLaVA: first all reference images, then the target frame
                    if ref_images_b64:
                        images_payload = ref_images_b64 + [target_b64]
                    else:
                        images_payload = [target_b64]

                    frame_observation = call_llava(images_payload, llava_prompt)
                    descriptions.append(frame_observation)

                    progress.progress(idx / total)

                # Optional: show raw frame-level observations
                with st.expander("Raw frame observations from LLaVA"):
                    for i, d in enumerate(descriptions, start=1):
                        st.markdown(f"**Frame {i}**")
                        st.text(d)
                        st.markdown("---")

                # Build summary prompt for Llama3
                summary_prompt = (
                    "You will receive brief observations from multiple frames of the SAME video of a child.\n"
                    "Each observation follows this structure: Relevance / Support / Assistance / Observation.\n\n"
                    "Your task is to write EXACTLY ONE short paragraph (4–6 sentences) addressed to a parent or caregiver.\n"
                    "You MUST clearly choose ONE of the following two recommendations:\n\n"
                    "A) If the observations show repeated or clear signs that COULD be compatible with muscle weakness or a neuromuscular condition "
                    "(for example: frequent use of arms/hands or external support to rise or keep balance, difficulty rising from the floor or a squat, "
                    "frequent falls, waddling gait, toe-walking, or unusual fatigue), then:\n"
                    '   - Start the paragraph with this sentence: "Based on the movements in these frames, I would recommend having your child evaluated by a healthcare professional." \n'
                    "   - Immediately follow with ONE sentence that begins with \"This is because\" or \"Because\", explaining in simple language the main movement patterns that raised concern "
                    "(for example repeated use of hands on thighs to stand up, needing support from furniture, or looking unsteady when walking).\n"
                    "   - Then add 1–3 short sentences explaining that an in-person exam with a healthcare profesional, can help clarify the situation.\n\n"
                    "B) If the observations do NOT consistently show those concerning signs and overall appear within a typical range for gross motor skills, then:\n"
                    '   - Start the paragraph with this sentence: "Based on the movements in these frames, there is no urgent sign that your child must see a specialist right away." \n'
                    "   - Immediately follow with ONE sentence that begins with \"This is because\" or \"Because\", summarizing in simple language that the movements mostly look stable and independent "
                    "without repeated strong use of arm support or obvious difficulty.\n"
                    "   - Then add 1–3 short sentences encouraging monitoring over time, normal play and practice, and mentioning that parents can still talk a healthcare profesional if worries remain.\n\n"
                    "In BOTH cases, end the paragraph with a clear sentence that this is NOT a diagnosis and is based only on limited video information.\n"
                    "Do NOT list the individual observations, do NOT mention frame counts or probabilities, and do NOT use bullet points or headings.\n"
                    "Output ONLY the paragraph.\n\n"
                    + "\n\n".join(descriptions)
            )

                summary = call_llama3(summary_prompt)

                st.subheader("Advice based on this video (not a diagnosis):")
                st.write(summary)


    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
