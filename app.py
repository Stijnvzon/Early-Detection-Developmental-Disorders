import streamlit as st
import cv2
import os
import base64
import requests
import json
import tempfile

# Titel van de app
st.title("Early detection of Developmental Disorders in Children")

# Upload video
uploaded_video = st.file_uploader("Upload a video of the child", type=["mp4", "mov", "avi"])

# Prompt gericht op motorische observatie
llava_prompt = (
    "You are observing a sequence of frames from a video showing a child."
    "Focus on motor movements that may indicate a developmental disorder or muscle weakness, such as using the arms for support when standing up."
    "Ignore clothing, background, and facial expression."
    "Describe only the physical actions and posture relevant for assessing motor skills and any signs of reduced strength."
    "Provide clear, objective descriptions for each frame without interpretation or judgment."
)
# llava_prompt = (
#     "Observeer de motorische bewegingen van het kind in deze afbeelding. "
#     "Let specifiek op signalen die kunnen wijzen op een ontwikkelingsstoornis, en die kunnen duiden op zwakkere spieren, "
#     "zoals steun zoeken met de armen bij het opstaan Negeer kleding, achtergrond en gezichtsuitdrukking. "
#     "Beschrijf alleen de fysieke handeling en houding die relevant zijn voor het beoordelen van motorische vaardigheden, en of er sprake is van krachtverlies."
# )

if uploaded_video is not None:
    # Tijdelijke opslag van video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(uploaded_video.read())
        video_path = tmp_video.name

    # Start analyse
    if st.button("Start Analyse"):
        st.info("Video is processing...")

        # Map om frames op te slaan
        frames_dir = tempfile.mkdtemp()
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = 2 * fps  # elke 2 seconden

        frame_count = 0
        saved_frames = []

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

        # Verzamel beschrijvingen van LLaVA
        descriptions = []
        for frame_file in saved_frames:
            with open(frame_file, "rb") as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            try:
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llava",
                        "prompt": llava_prompt,
                        "images": [image_base64]
                    },
                    stream=True
                )

                description = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            part = line.decode("utf-8")
                            json_part = json.loads(part)
                            description += json_part.get("response", "")
                        except Exception as e:
                            st.warning(f"Fout bij verwerken van response: {e}")
                            continue

                # st.subheader(f"Beschrijving voor frame {os.path.basename(frame_file)}")
                # st.write(description)
                descriptions.append(description)

            except requests.exceptions.RequestException as e:
                st.error(f"Fout bij verzenden van verzoek voor {frame_file}: {e}")

        # Samenvattingsprompt voor LLaMA3
        summary_prompt = (
            "You will receive descriptions of multiple frames from a video of a child, presented in chronological order."
            "Analyze these descriptions with a focus on motor movements that may indicate a developmental disorder or muscle weakness, such as using the arms for support when standing up."
            "Create a concise summary of any indications that point to a possible developmental delay—this should be a general summary for the entire video, not per frame."
            "Finally, provide advice on whether consulting a doctor is recommended based on the summary."
            "When giving advice, include a reassuring note that every child develops at their own pace and that these observations alone are not a definitive diagnosis."
            "\n\n"
            + "\n\n".join(descriptions)
        )

        # summary_prompt = (
        #     "Je krijgt beschrijvingen van video-frames van een kind, in chronologische volgorde. "
        #     "Analyseer deze beschrijvingen met focus op motorische bewegingen die kunnen duiden op een ontwikkelingsstoornis. "
        #     "Signalen die erg belangrijk zijn, zijn signalen zoals steun zoeken met de armen bij het opstaan. Dit soort signalen kunnen duiden op zwakkere spieren."
        #     "Maak een samenvatting in het Nederlands van de aanwijzingen die er zijn voor een mogelijke ontwikkelingsachterstand, dit is een algemene samenvatting, niet per frame."
        #     "Geef als laatste een advies over het wel of niet bezoeken van een doctor op basis van de samenvatting."
        #     "\n\n"
        #     + "\n\n".join(descriptions)
        # )

        # Verstuur samenvattingsprompt naar LLaMA3
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3",
                    "prompt": summary_prompt
                },
                stream=True
            )

            summary = ""
            for line in response.iter_lines():
                if line:
                    try:
                        part = line.decode("utf-8")
                        json_part = json.loads(part)
                        summary += json_part.get("response", "")
                    except Exception as e:
                        st.warning(f"Fout bij verwerken van samenvatting: {e}")
                        continue

            st.subheader("Summary of development of the child:")
            st.write(summary)

        except requests.exceptions.RequestException as e:
            st.error(f"Fout bij verzenden van samenvattingsverzoek: {e}")
