# YourMT3+ Instrument Conditioning Implementation

## Overview

This implementation adds instrument-specific transcription capabilities to YourMT3+ to address the problem of inconsistent instrument classification during transcription. The main issues addressed are:

1. **Instrument switching mid-track**: Model switches between instruments (e.g., vocals → violin → guitar) on single-instrument audio
2. **Poor instrument-specific transcription**: Incomplete transcription of specific instruments (e.g., saxophone solo, flute parts)
3. **Lack of user control**: No way to specify which instrument you want transcribed

## Implementation Details

### 1. Core Architecture Changes

#### **model_helper.py** - Enhanced transcription function
- Added `instrument_hint` parameter to `transcribe()` function
- New `create_instrument_task_tokens()` function that leverages YourMT3's existing task conditioning system
- New `filter_instrument_consistency()` function for post-processing filtering

#### **app.py** - Enhanced Gradio Interface 
- Added instrument selection dropdown with options:
  - Auto (detect all instruments) 
  - Vocals/Singing
  - Guitar, Piano, Violin, Bass
  - Drums, Saxophone, Flute
- Updated both "Upload audio" and "From YouTube" tabs
- Maintains backward compatibility with existing functionality

#### **transcribe_cli.py** - New Command Line Interface
- Standalone CLI tool with full instrument conditioning support
- Support for confidence thresholds and filtering options
- Verbose output and error handling

### 2. How It Works

#### **Task Token Conditioning**
The implementation leverages YourMT3's existing task conditioning system:

```python
# Maps instrument hints to task events
instrument_mapping = {
    'vocals': 'transcribe_singing',
    'drums': 'transcribe_drum', 
    'guitar': 'transcribe_all'  # falls back to general transcription
}
```

#### **Post-Processing Consistency Filtering**
When an instrument hint is provided, the system:

1. Analyzes the transcribed notes to identify the dominant instrument
2. Filters out notes from other instruments if confidence is above threshold
3. Converts remaining notes to the target instrument program

```python
def filter_instrument_consistency(pred_notes, confidence_threshold=0.7):
    # Count instrument occurrences
    # If dominant instrument > threshold, filter others
    # Convert notes to primary instrument
```

## Usage Examples

### 1. Gradio Web Interface

1. **Upload audio tab**: 
   - Upload your audio file
   - Select target instrument from dropdown
   - Click "Transcribe"

2. **YouTube tab**:
   - Paste YouTube URL
   - Select target instrument  
   - Click "Get Audio from YouTube" then "Transcribe"

### 2. Command Line Interface

```bash
# Basic transcription (all instruments)
python transcribe_cli.py audio.wav

# Transcribe vocals only
python transcribe_cli.py audio.wav --instrument vocals

# Force single instrument with high confidence threshold
python transcribe_cli.py audio.wav --single-instrument --confidence-threshold 0.9

# Transcribe guitar with verbose output
python transcribe_cli.py guitar_solo.wav --instrument guitar --verbose

# Custom output path
python transcribe_cli.py audio.wav --instrument piano --output my_piano.mid
```

### 3. Python API Usage

```python
from model_helper import load_model_checkpoint, transcribe

# Load model
model = load_model_checkpoint(args=model_args, device="cuda")

# Prepare audio info
audio_info = {
    "filepath": "audio.wav",
    "track_name": "my_audio"
}

# Transcribe with instrument hint
midi_file = transcribe(model, audio_info, instrument_hint="vocals")
```

## Supported Instruments

- **vocals**, **singing**, **voice** → Uses existing 'transcribe_singing' task
- **drums**, **drum**, **percussion** → Uses existing 'transcribe_drum' task  
- **guitar**, **piano**, **violin**, **bass**, **saxophone**, **flute** → Uses enhanced filtering with 'transcribe_all' task

## Technical Benefits

### 1. **Leverages Existing Architecture**
- Uses YourMT3's built-in task conditioning system
- No model retraining required
- Backward compatible with existing code

### 2. **Two-Stage Approach**
- **Stage 1**: Task token conditioning biases the model toward specific instruments
- **Stage 2**: Post-processing filtering ensures consistency

### 3. **Configurable Confidence**
- Adjustable confidence thresholds for filtering
- Balances between accuracy and completeness

## Limitations & Future Improvements

### Current Limitations
1. **Limited task tokens**: Only vocals and drums have dedicated task tokens
2. **Post-processing dependency**: Other instruments rely on filtering 
3. **No instrument-specific training**: Uses general model weights

### Future Improvements
1. **Extended task vocabulary**: Add dedicated task tokens for more instruments
2. **Instrument-specific models**: Train specialized decoders for each instrument
3. **Confidence scoring**: Add per-note confidence scores for better filtering
4. **Pitch-based filtering**: Use pitch ranges typical for each instrument

## Installation & Setup

1. **Install dependencies** (from existing YourMT3 requirements):
   ```bash
   pip install torch torchaudio transformers gradio
   ```

2. **Model weights**: Ensure YourMT3 model weights are in `amt/logs/`

3. **Run web interface**:
   ```bash
   python app.py
   ```

4. **Run CLI**:
   ```bash
   python transcribe_cli.py --help
   ```

## Testing

Run the test suite:
```bash
python test_instrument_conditioning.py
```

This will verify:
- Code syntax and imports
- Function availability  
- Basic functionality (when dependencies are available)

## Conclusion

This implementation provides a practical solution to YourMT3+'s instrument confusion problem by:

1. **Adding user control** over instrument selection
2. **Leveraging existing architecture** for minimal changes
3. **Providing multiple interfaces** (web, CLI, API)
4. **Maintaining backward compatibility**

The approach addresses the core issue you mentioned: "*so many times i upload vocals and it transcribes half right, as vocals, then switches to violin although the whole track is just vocals*" by giving you direct control over the transcription focus.
