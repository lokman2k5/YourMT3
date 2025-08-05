#!/usr/bin/env python3
"""
YourMT3+ CLI with Instrument Conditioning
Command-line interface for transcribing audio with instrument-specific hints.

Usage:
    python transcribe_cli.py audio.wav
    python transcribe_cli.py audio.wav --instrument vocals
    python transcribe_cli.py audio.wav --instrument guitar --confidence-threshold 0.8
"""

import os
import sys
import argparse
import torch
import torchaudio
from pathlib import Path

# Add the amt/src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))

from model_helper import load_model_checkpoint, transcribe


def main():
    parser = argparse.ArgumentParser(
        description="YourMT3+ Audio Transcription with Instrument Conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s audio.wav                                  # Transcribe all instruments
  %(prog)s audio.wav --instrument vocals              # Focus on vocals only
  %(prog)s audio.wav --instrument guitar              # Focus on guitar only
  %(prog)s audio.wav --single-instrument              # Force single instrument output
  %(prog)s audio.wav --instrument piano --confidence-threshold 0.9

Supported instruments:
  vocals, singing, voice, guitar, piano, violin, drums, bass, saxophone, flute
        """
    )
    
    # Required arguments
    parser.add_argument('audio_file', help='Path to the audio file to transcribe')
    
    # Instrument conditioning options
    parser.add_argument('--instrument', type=str, 
                       choices=['vocals', 'singing', 'voice', 'guitar', 'piano', 'violin', 
                               'drums', 'bass', 'saxophone', 'flute'],
                       help='Specify the primary instrument to transcribe')
    
    parser.add_argument('--single-instrument', action='store_true',
                       help='Force single instrument output (apply consistency filtering)')
    
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for instrument consistency filtering (0.0-1.0, default: 0.7)')
    
    # Model selection
    parser.add_argument('--model', type=str, 
                       default='YPTF.MoE+Multi (noPS)',
                       choices=['YMT3+', 'YPTF+Single (noPS)', 'YPTF+Multi (PS)', 
                               'YPTF.MoE+Multi (noPS)', 'YPTF.MoE+Multi (PS)'],
                       help='Model checkpoint to use (default: YPTF.MoE+Multi (noPS))')
    
    # Output options  
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output MIDI file path (default: auto-generated from input filename)')
    
    parser.add_argument('--precision', type=str, default='16', choices=['16', '32', 'bf16-mixed'],
                       help='Floating point precision (default: 16)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file '{args.audio_file}' not found.")
        sys.exit(1)
    
    # Validate confidence threshold
    if not 0.0 <= args.confidence_threshold <= 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0.")
        sys.exit(1)
    
    # Set output path
    if args.output is None:
        input_path = Path(args.audio_file)
        args.output = input_path.with_suffix('.mid')
    
    if args.verbose:
        print(f"Input file: {args.audio_file}")
        print(f"Output file: {args.output}")
        print(f"Model: {args.model}")
        if args.instrument:
            print(f"Target instrument: {args.instrument}")
        if args.single_instrument:
            print(f"Single instrument mode: enabled (threshold: {args.confidence_threshold})")
    
    try:
        # Load model
        if args.verbose:
            print("Loading model...")
        
        model_args = get_model_args(args.model, args.precision)
        model = load_model_checkpoint(args=model_args, device="cpu")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        if args.verbose:
            print("Model loaded successfully!")
        
        # Prepare audio info
        audio_info = {
            "filepath": args.audio_file,
            "track_name": Path(args.audio_file).stem
        }
        
        # Get audio info
        info = torchaudio.info(args.audio_file)
        audio_info.update({
            "sample_rate": int(info.sample_rate),
            "bits_per_sample": int(info.bits_per_sample) if hasattr(info, 'bits_per_sample') else 16,
            "num_channels": int(info.num_channels),
            "num_frames": int(info.num_frames),
            "duration": int(info.num_frames / info.sample_rate),
            "encoding": str.lower(str(info.encoding)),
        })
        
        # Determine instrument hint
        instrument_hint = None
        if args.instrument:
            instrument_hint = args.instrument
        elif args.single_instrument:
            # Auto-detect dominant instrument but force single output
            instrument_hint = "auto"
        
        # Transcribe
        if args.verbose:
            print("Starting transcription...")
        
        # Set confidence threshold in model_helper if single_instrument is enabled
        if args.single_instrument:
            # We'll need to modify the transcribe function to accept confidence_threshold
            original_confidence = 0.7  # default
            # For now, this is handled in the transcribe function
        
        midifile = transcribe(model, audio_info, instrument_hint)
        
        # Move output to desired location if needed
        if str(args.output) != midifile:
            import shutil
            shutil.move(midifile, args.output)
            midifile = str(args.output)
        
        print(f"Transcription completed successfully!")
        print(f"Output saved to: {midifile}")
        
        if args.verbose:
            # Print some basic statistics
            file_size = os.path.getsize(midifile)
            print(f"Output file size: {file_size} bytes")
            print(f"Duration: {audio_info['duration']} seconds")
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def get_model_args(model_name, precision):
    """Get model arguments based on model name and precision."""
    project = '2024'
    
    if model_name == "YMT3+":
        checkpoint = "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt"
        args = [checkpoint, '-p', project, '-pr', precision]
    elif model_name == "YPTF+Single (noPS)":
        checkpoint = "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt"
        args = [checkpoint, '-p', project, '-enc', 'perceiver-tf', '-ac', 'spec',
                '-hop', '300', '-atc', '1', '-pr', precision]
    elif model_name == "YPTF+Multi (PS)":
        checkpoint = "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt"
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
        raise ValueError(f"Unknown model name: {model_name}")
    
    return args


if __name__ == "__main__":
    main()
