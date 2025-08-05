#!/usr/bin/env python3
"""
Test script for YourMT3+ instrument conditioning features.
This script tests the new instrument-specific transcription capabilities.
"""

import os
import sys
import subprocess
from pathlib import Path

def test_cli():
    """Test the CLI interface with different instrument hints."""
    
    # Use an example audio file
    test_audio = "/home/lyzen/Downloads/YourMT3/examples/mirst493.wav"
    
    if not os.path.exists(test_audio):
        print(f"Test audio file not found: {test_audio}")
        return False
    
    print("Testing YourMT3+ CLI with instrument conditioning...")
    print(f"Test audio: {test_audio}")
    
    # Test cases
    test_cases = [
        {
            "name": "Default (all instruments)",
            "args": [test_audio],
            "expected_output": "mirst493.mid"
        },
        {
            "name": "Vocals only",
            "args": [test_audio, "--instrument", "vocals", "--verbose"],
            "expected_output": "mirst493.mid"
        },
        {
            "name": "Single instrument mode",
            "args": [test_audio, "--single-instrument", "--confidence-threshold", "0.8", "--verbose"],
            "expected_output": "mirst493.mid"
        }
    ]
    
    cli_script = "/home/lyzen/Downloads/YourMT3/transcribe_cli.py"
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        
        # Clean up previous output
        output_file = test_case['expected_output']
        if os.path.exists(output_file):
            os.remove(output_file)
        
        # Run the CLI command 
        cmd = ["python", cli_script] + test_case['args']
        print(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                print("✓ Command executed successfully")
                print("STDOUT:", result.stdout)
                
                if os.path.exists(output_file):
                    print(f"✓ Output file created: {output_file}")
                    file_size = os.path.getsize(output_file)
                    print(f"  File size: {file_size} bytes")
                else:
                    print(f"✗ Expected output file not found: {output_file}")
            else:
                print(f"✗ Command failed with return code {result.returncode}")
                print("STDERR:", result.stderr)
                print("STDOUT:", result.stdout)
                
        except subprocess.TimeoutExpired:
            print("✗ Command timed out after 5 minutes")
        except Exception as e:
            print(f"✗ Error running command: {e}")
    
    print("\n" + "="*50)
    print("CLI Test completed!")


def test_gradio_interface():
    """Test the Gradio interface updates."""
    print("\n--- Testing Gradio Interface Updates ---")
    
    try:
        # Import the updated app to check for syntax errors
        sys.path.append("/home/lyzen/Downloads/YourMT3")
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("app", "/home/lyzen/Downloads/YourMT3/app.py")
        app_module = importlib.util.module_from_spec(spec)
        
        print("✓ app.py imports successfully")
        
        # Check if our new functions exist
        spec.loader.exec_module(app_module)
        
        if hasattr(app_module, 'process_audio'):
            print("✓ process_audio function found")
        else:
            print("✗ process_audio function not found")
            
        print("✓ Gradio interface syntax check passed")
        
    except Exception as e:
        print(f"✗ Gradio interface test failed: {e}")
        import traceback
        traceback.print_exc()


def test_model_helper():
    """Test the model_helper updates."""
    print("\n--- Testing Model Helper Updates ---")
    
    try:
        sys.path.append("/home/lyzen/Downloads/YourMT3")
        sys.path.append("/home/lyzen/Downloads/YourMT3/amt/src")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("model_helper", "/home/lyzen/Downloads/YourMT3/model_helper.py")
        model_helper = importlib.util.module_from_spec(spec)
        
        print("✓ model_helper.py imports successfully")
        
        # Check if our new functions exist
        spec.loader.exec_module(model_helper)
        
        if hasattr(model_helper, 'create_instrument_task_tokens'):
            print("✓ create_instrument_task_tokens function found")
        else:
            print("✗ create_instrument_task_tokens function not found")
            
        if hasattr(model_helper, 'filter_instrument_consistency'):
            print("✓ filter_instrument_consistency function found")
        else:
            print("✗ filter_instrument_consistency function not found")
            
        print("✓ Model helper syntax check passed")
        
    except Exception as e:
        print(f"✗ Model helper test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("YourMT3+ Instrument Conditioning Test Suite")
    print("=" * 50)
    
    # Test individual components
    test_model_helper()
    test_gradio_interface()
    
    # Uncomment this to test the full CLI (requires model weights)
    # test_cli()
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nTo test the full functionality:")
    print("1. Ensure model weights are available in amt/logs/")
    print("2. Run: python transcribe_cli.py examples/mirst493.wav --instrument vocals")
    print("3. Or run the Gradio interface: python app.py")
