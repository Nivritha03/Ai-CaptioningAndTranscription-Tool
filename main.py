import assemblyai as aai
import gradio as gr
import yt_dlp as ytdl
from pydub import AudioSegment
import os
import tempfile
import cv2

# Initialize AssemblyAI with your API key
aai.settings.api_key = "f65a0152fbea41c9bc6c2ca23c9e497b"

def extract_audio_with_pydub(video_path, audio_path="temp_audio.wav"):
    """Extract audio from a video file and save it as a WAV file."""
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(16000)
    audio.export(audio_path, format="wav")
    return audio_path

def transcribe_audio_or_video(file_path, callback=None):
    """Transcribe audio using AssemblyAI."""
    if file_path.endswith(('.mp4', '.mkv', '.avi')):
        audio_path = extract_audio_with_pydub(file_path)
        file_path = audio_path

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(file_path)
    
    if callback:
        callback(transcript.text)
    return transcript.text

def download_online_media(url, save_path):
    """Download and process media file from URL in real-time."""
    output_dir = os.path.dirname(save_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    
    ydl_opts = {
        'format': 'best[ext=mp4]',  # Prefer MP4 format for better compatibility
        'outtmpl': output_template,
        'progress_hooks': [lambda d: print(f"Downloading: {d['_percent_str']}")],
        'merge_output_format': 'mp4',  # Force MP4 output
        'keepvideo': True,
        'writethumbnail': False
    }
    
    with ytdl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info['title']
        # Get the actual downloaded file path with mp4 extension
        downloaded_file = os.path.join(output_dir, f"{title}.mp4")
        print(f"Video file path: {downloaded_file}")
        return downloaded_file, downloaded_file  # Return local path for both

def analyze_video_with_opencv(video_path):
    """Extract basic information from a video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return num_frames, fps, width, height

def enhanced_process_input(file_input, url_input, is_url):
    try:
        if is_url:
            if not url_input or not url_input.startswith(('http://', 'https://')):
                return None, "Error: Please provide a valid URL."
            
            file_path, video_path = download_online_media(url_input, tempfile.mktemp())
            transcript = transcribe_audio_or_video(file_path)
            return video_path, transcript
        else:
            if not file_input:
                return None, "Error: Please upload a file."
            
            transcript = transcribe_audio_or_video(file_input.name)
            return file_input.name, transcript

    except Exception as e:
        return None, f"Error: {str(e)}"

# Build Gradio Interface
with gr.Blocks() as app:
    gr.Markdown("# Real-time Media Transcription with Captions")

    with gr.Row():
        file_input = gr.File(label="Upload Audio/Video")
        url_input = gr.Textbox(label="Enter URL (Online Media)")
        url_checkbox = gr.Checkbox(label="Is this a URL?", value=False)

    with gr.Row():
        video_output = gr.Video(label="Media Player")
        transcript_output = gr.Textbox(label="Transcription", interactive=True)

    submit_btn = gr.Button("Process")
    submit_btn.click(
        enhanced_process_input,
        inputs=[file_input, url_input, url_checkbox],
        outputs=[video_output, transcript_output]
    )

if __name__ == "__main__":
    app.launch()
