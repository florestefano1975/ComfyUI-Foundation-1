# ComfyUI-Foundation-1

An unofficial custom node for **ComfyUI** to run the **Foundation-1** music generation model by RoyalCities. This node allows for high-fidelity text-to-audio generation directly within your ComfyUI workflows, featuring precise duration control and optimized local model management.

## 🚀 Features
* **Local Loading**: Load .safetensors and .json configs directly from your models folder (no internet required after setup).
* **Precision Duration**: Generates audio exactly to the requested length (no more silence padding at the end).
* **VRAM Optimized**: Dynamically adjusts generation size to save memory and speed up processing for shorter clips.
* **Musical Control**: Full support for BPM, Key, Scale, and Timbre descriptors.

---

## 🛠️ Installation

### 1. Clone the repository
Navigate to your ComfyUI custom_nodes folder and clone this repo:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/florestefano1975/ComfyUI-Foundation-1
```

### 2. Environment Setup (ComfyUI Windows Portable)
This model requires specific libraries that can be tricky on newer Python versions (3.12/3.13). Open a terminal in your python_embeded folder and run:

```bash
# 1. Update core tools
.\python.exe -m pip install --upgrade pip setuptools wheel

# 2. Install Stable Audio Tools without forcing problematic dependencies
.\python.exe -m pip install stable-audio-tools --no-deps

# 3. Install remaining requirements with binary preference (avoids C++ compilation errors)
.\python.exe -m pip install -r ..\ComfyUI\custom_nodes\ComfyUI-Foundation-1\requirements.txt --prefer-binary
```

> **Note**: If you see ModuleNotFoundError: No module named 'pkg_resources', run:
> .\python.exe -m pip install setuptools==70.0.0

---

## 📂 Model Setup

1.  Download the **Foundation-1** weights from [Hugging Face](https://huggingface.co/RoyalCities/Foundation-1).
2.  Create the following folder in your ComfyUI directory: ComfyUI/models/foundation-1/.
3.  Place your files there. **Crucial**: Both files must share the same filename:
    * Foundation_1.safetensors
    * Foundation_1.json (This is the model_config.json from the HF repo).

---

## 🎹 Usage

### Nodes
* **F1 Local Model Loader**: Select your model from the dropdown. It automatically detects the matching JSON config in the same folder.
* **F1 Audio Sampler**:
    * **prompt**: Use descriptive tags (e.g., Techno, 128 BPM, Dark, Industrial Synth).
    * **seconds_total**: Desired output length.
    * **steps**: Diffusion steps (50-100 recommended for quality).
    * **cfg_scale**: Guidance scale (7.0 is usually ideal).

### Prompting Tips
Foundation-1 thrives on technical descriptors. Try this structure:
[Instrument], [Genre], [BPM], [Key/Scale], [Timbre], [Effects]

*Example:* Aggressive Bassline, Cyberpunk, 100 BPM, D Minor, Distorted, Reverb

---

## ⚠️ Troubleshooting
* **Silent output?**: The node now automatically crops silence. If you still hear nothing, check if your seconds_total is within the model's capacity (~23s).
* **Cuda Out of Memory**: Lower the seconds_total or reduce the steps.
* **Pandas/Numpy Errors**: Ensure you followed the "Environment Setup" steps above to bypass the legacy dependency conflicts of Stable Audio Tools.

---

## Credits
* **Model by**: [RoyalCities](https://huggingface.co/RoyalCities/Foundation-1)
* **Architecture**: [Stable Audio Tools](https://github.com/Stability-AI/stable-audio-tools) by Stability AI.
