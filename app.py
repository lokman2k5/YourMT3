import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))


import glob
import gradio as gr

from gradio_helper import *
from model_helper import *

# @title Load Checkpoint
model_name = 'YPTF.MoE+Multi (noPS)' # @param ["YMT3+", "YPTF+Single (noPS)", "YPTF+Multi (PS)", "YPTF.MoE+Multi (noPS)", "YPTF.MoE+Multi (PS)"]
precision = '16' # @param ["32", "bf16-mixed", "16"]
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



AUDIO_EXAMPLES = glob.glob('/content/examples/*.*', recursive=True)
YOUTUBE_EXAMPLES = ["https://www.youtube.com/watch?v=vMboypSkj3c"]

theme = 'gradio/dracula_revamped' #'Insuz/Mocha' #gr.themes.Soft()
with gr.Blocks(theme=theme) as demo:

    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown(
            """
            ### YourMT3+: Multi-instrument Music Transcription with Enhanced Transformer Architectures and Cross-dataset Stem Augmentation
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
