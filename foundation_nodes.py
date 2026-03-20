import os
import torch
import folder_paths
from stable_audio_tools import create_model_from_config
from stable_audio_tools.inference.generation import generate_diffusion_cond
from safetensors.torch import load_file

models_dir = os.path.join(folder_paths.models_dir, "foundation-1")
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

folder_paths.add_model_folder_path("foundation-1", models_dir)

class Foundation1LocalLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_filename": (folder_paths.get_filename_list("foundation-1"), ),
            }
        }

    RETURN_TYPES = ("F1_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_local_model"
    CATEGORY = "Foundation1"

    def load_local_model(self, model_filename):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = folder_paths.get_full_path("foundation-1", model_filename)
        
        config_path = model_path.replace(".safetensors", ".json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Manca il file di configurazione JSON in: {config_path}")

        print(f"Caricamento locale di Foundation-1: {model_path}")
        
        import json
        with open(config_path, "r") as f:
            model_config = json.load(f)
        
        model = create_model_from_config(model_config)
        
        sd = load_file(model_path)
        model.load_state_dict(sd)
        
        model = model.to(device).eval()
        
        return ({"model": model, "config": model_config, "device": device},)

class Foundation1Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_data": ("F1_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "Techno, 128 BPM, Dark, Industrial"}),
                "seconds_total": ("INT", {"default": 8, "min": 1, "max": 20}),
                "steps": ("INT", {"default": 50, "min": 10, "max": 250}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 20.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "generate_audio"
    CATEGORY = "Foundation1"

    def generate_audio(self, model_data, prompt, seconds_total, steps, cfg_scale, seed):
        model = model_data["model"]
        device = model_data["device"]
        sample_rate = model_data["config"]["sample_rate"]
        
        requested_samples = int(seconds_total * sample_rate)

        sample_size = (requested_samples + 1023) // 1024 * 1024
        
        max_model_samples = model_data["config"].get("sample_size", sample_size)
        if sample_size > max_model_samples:
            print(f"Attenzione: Richiesti {sample_size} campioni, ma il modello supporta massimo {max_model_samples}. Limito la generazione.")
            sample_size = max_model_samples

        conditioning = [{
            "prompt": prompt,
            "seconds_start": 0,
            "seconds_total": seconds_total
        }]

        print(f"Richiesti: {seconds_total}s ({requested_samples} campioni)")
        print(f"Configurazione Sampler (multiplo 1024): {sample_size} campioni")

        output = generate_diffusion_cond(
            model,
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-2m",
            device=device,
            seed=seed
        )

        waveform = output.squeeze(0).cpu()

        if waveform.shape[1] > requested_samples:
            waveform = waveform[:, :requested_samples]
            print(f"Pulizia finale eseguita. Output finale: {waveform.shape[1]} campioni.")

        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

NODE_CLASS_MAPPINGS = {
    "Foundation1LocalLoader": Foundation1LocalLoader,
    "Foundation1Sampler": Foundation1Sampler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Foundation1LocalLoader": "F1 Local Model Loader",
    "Foundation1Sampler": "F1 Audio Sampler"
}