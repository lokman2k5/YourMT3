import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))

import subprocess
from typing import Tuple, Dict, Literal
from ctypes import ArgumentError

from html_helper import *
from model_helper import *

from pytube import YouTube
import torch
import torchaudio
import glob
import gradio as gr



# @title Load Checkpoint
model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
precision = '16' if torch.cuda.is_available() else '32'# @param ["32", "bf16-mixed", "16"]
project = '2024'

if model_name == "YMT3+":
    checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
    args = [checkpoint, '-p', project, '-pr', precision]
elif model_name == "YPTF+Single (noPS)":
    checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
    args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
            '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256',
            '-dec', 'multi-t5', '-nl', '26', '-enc', 'perceiver-tf',
            '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (noPS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
elif model_name == "YPTF.MoE+Multi (PS)":
    checkpoint = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
    args = [checkpoint, '-p', project, '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', precision]
else:
    raise ValueError(model_name)

model = load_model_checkpoint(args=args)

# @title GradIO helper


def prepare_media(source_path_or_url: os.PathLike,
                  source_type: Literal['audio_filepath', 'youtube_url'],
                  delete_video: bool = True) -> Dict:
    """prepare media from source path or youtube, and return audio info"""
    # Get audio_file
    if source_type == 'audio_filepath':
        audio_file = source_path_or_url
    elif source_type == 'youtube_url':
        # Download from youtube
        try:
            # Try PyTube first
            yt = YouTube(source_path_or_url)
            audio_stream = min(yt.streams.filter(only_audio=True), key=lambda s: s.bitrate)
            mp4_file = audio_stream.download(output_path='downloaded') # ./downloaded
            audio_file = mp4_file[:-3] + 'mp3'
            subprocess.run(['ffmpeg', '-i', mp4_file, '-ac', '1', audio_file])
            os.remove(mp4_file)
        except Exception as e:
            try:
                # Try alternative
                print(f"Failed with PyTube, error: {e}. Trying yt-dlp...")
                audio_file = './downloaded/yt_audio'
                subprocess.run(['yt-dlp', '-x', source_path_or_url, '-f', 'bestaudio',
                    '-o', audio_file, '--audio-format', 'mp3', '--restrict-filenames',
                    '--force-overwrites'])
                audio_file += '.mp3'
            except Exception as e:
                print(f"Alternative downloader failed, error: {e}. Please try again later!")
                return None
    else:
        raise ValueError(source_type)

    # Create info
    info = torchaudio.info(audio_file)
    return {
        "filepath": audio_file,
        "track_name": os.path.basename(audio_file).split('.')[0],
        "sample_rate": int(info.sample_rate),
        "bits_per_sample": int(info.bits_per_sample),
        "num_channels": int(info.num_channels),
        "num_frames": int(info.num_frames),
        "duration": int(info.num_frames / info.sample_rate),
        "encoding": str.lower(info.encoding),
        }

def process_audio(audio_filepath):
    if audio_filepath is None:
        return None
    audio_info = prepare_media(audio_filepath, source_type='audio_filepath')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

def process_video(youtube_url):
    if 'youtu' not in youtube_url:
        return None
    audio_info = prepare_media(youtube_url, source_type='youtube_url')
    midifile = transcribe(model, audio_info)
    midifile = to_data_url(midifile)
    return create_html_from_midi(midifile) # html midiplayer

def play_video(youtube_url):
    if 'youtu' not in youtube_url:
        return None
    return create_html_youtube_player(youtube_url)



AUDIO_EXAMPLES = glob.glob('examples/*.*', recursive=True)
YOUTUBE_EXAMPLES = ["https://www.youtube.com/watch?v=vMboypSkj3c"]

# theme = 'gradio/dracula_revamped' #'Insuz/Mocha' #gr.themes.Soft()
# with gr.Blocks(theme=theme) as demo:
theme = gr.Theme.from_hub("gradio/dracula_revamped")
theme.text_md = '9px'
theme.text_lg = '11px'
with gr.Blocks(theme=theme) as demo:

    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
            """
            ## 🎶YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation
            #### Caution:
            - Running on CPU takes more than 3 minutes for a 30-second input, and it takes less than 10s with T4 GPU-small ($0.4/hr).
            - For acadmic reproduction purpose, we strongly recommend to use  or [Colab Demo](https://colab.research.google.com/drive/1AgOVEBfZknDkjmSRA7leoa81a2vrnhBG?usp=sharing) with multiple checkpoints.
            ### [Paper](https://arxiv.org/abs/2407.04822) [Code](https://github.com/mimbres/YourMT3)
            """)

    with gr.Group():
        with gr.Tab("Upload audio"):
            # Input
            audio_input = gr.Audio(label="Record Audio", type="filepath",
                                show_share_button=True, show_download_button=True)
            # Display examples
            gr.Examples(examples=AUDIO_EXAMPLES, inputs=audio_input)
            # Submit button
            transcribe_audio_button = gr.Button("Transcribe", variant="primary")
            # Transcribe
            output_tab1 = gr.HTML()
            # audio_output = gr.Text(label="Audio Info")
            # transcribe_audio_button.click(process_audio, inputs=audio_input, outputs=output_tab1)
            transcribe_audio_button.click(process_audio, inputs=audio_input, outputs=output_tab1)

        with gr.Tab("From YouTube"):
            with gr.Row():
                # Input URL
                youtube_url = gr.Textbox(label="YouTube Link URL",
                        placeholder="https://youtu.be/...")
                # Play youtube
                youtube_player = gr.HTML(render=True)
            with gr.Row():
                # Play button
                play_video_button = gr.Button("Play", variant="primary")
                # Submit button
                transcribe_video_button = gr.Button("Transcribe", variant="primary")
            # Transcribe
            output_tab2 = gr.HTML(render=True)
            # video_output = gr.Text(label="Video Info")
            transcribe_video_button.click(process_video, inputs=youtube_url, outputs=output_tab2)
            # Play
            play_video_button.click(play_video, inputs=youtube_url, outputs=youtube_player)

            # Display examples
            gr.Examples(examples=YOUTUBE_EXAMPLES, inputs=youtube_url)

demo.launch(debug=True)
