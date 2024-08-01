import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))


import glob
import gradio as gr

from gradio_helper import *

AUDIO_EXAMPLES = glob.glob('/content/examples/*.*', recursive=True)
YOUTUBE_EXAMPLES = ["https://www.youtube.com/watch?v=vMboypSkj3c"]

theme = 'gradio/dracula_revamped' #'Insuz/Mocha' #gr.themes.Soft()
with gr.Blocks(theme=theme) as demo:

    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
            """
            ## YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation
            ### Sungkyun Chang, Emmanouil Benetos, Holger Kirchhoff and Simon Dixon, IEEE MLSP 2024 (to appear)
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
