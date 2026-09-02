import os
import random
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

import gc

import gradio as gr
from gradio import Brush, Eraser
from PIL import Image, ImageEnhance, ImageFilter, ImageOps



ROOT = Path(__file__).parent
SD_BIN_DIR = ROOT / "sd_bin"
MODEL_CONFIG_PATH = ROOT / "models" / "zimage" / "selected_model.txt"
MODEL_NAME = os.environ.get("ZIMAGE_MODEL_NAME")
if not MODEL_NAME and MODEL_CONFIG_PATH.exists():
    MODEL_NAME = MODEL_CONFIG_PATH.read_text(encoding="utf-8").strip()
if not MODEL_NAME:
    MODEL_NAME = "z_image_turbo_Q4_0.gguf"
MODEL_PATH = str(ROOT / "models" / "zimage" / MODEL_NAME)
LORA_DIR = ROOT / "models" / "loras"
OUTDIR = ROOT / "outputs"
TEMP_INPUT_DIR = OUTDIR / "_tmp_inputs"

DEFAULT_VAE_PATH = str(ROOT / "models" / "vae" / "ae.safetensors")
DEFAULT_LLM_PATH = str(ROOT / "models" / "llm" / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf")

OUTDIR.mkdir(exist_ok=True)
LORA_DIR.mkdir(exist_ok=True)
TEMP_INPUT_DIR.mkdir(exist_ok=True)

current_proc = None
current_job_id = None
generation_jobs = []
generation_lock = threading.RLock()
worker_lock = threading.Lock()
generation_worker_thread = None
stop_requested = False
latest_image = None
latest_status = "Ready."
latest_time = "Generation Time: **0s**"
latest_command = ""
FIRST_RUN = True
LAST_SEED = None
LAST_IMG2IMG_SEED = None
LAST_INPAINT_SEED = None

RES_PRESETS = [
    ("1:1 (256x256)", 256, 256),
    ("1:1 (512x512)", 512, 512),
    ("1:1 (768x768)", 768, 768),
    ("1:1 (1024x1024)", 1024, 1024),
    ("16:9 (640x384)", 640, 384),
    ("16:9 (896x512)", 896, 512),
    ("16:9 (1024x576)", 1024, 576),
    ("9:16 (384x640)", 384, 640),
    ("9:16 (512x896)", 512, 896),
    ("9:16 (576x1024)", 576, 1024),
    ("4:3 (640x480)", 640, 480),
    ("4:3 (768x576)", 768, 576),
    ("3:2 (768x512)", 768, 512),
    ("2:3 (512x768)", 512, 768),
]

SIZE_OPTIONS = sorted({s for _, w, h in RES_PRESETS for s in (w, h)})
VRAM_PRESETS = [
    "4GB (safest)",
    "6-8GB (balanced)",
    "10GB+ (fastest)",
]
LORA_APPLY_MODES = ["auto", "immediately", "at_runtime"]
TXT2IMG_PROMPTS = {
    "Portrait": "cinematic portrait of a woman in a red dress inside a cozy cafe, soft window light, natural skin texture, 35mm photo",
    "Product": "premium product photo of a matte black wireless speaker on a clean desk, softbox lighting, sharp details, commercial photography",
    "Landscape": "wide landscape photo of a misty mountain valley at sunrise, dramatic light, realistic atmosphere, high detail",
    "Fantasy": "fantasy castle above a glowing forest, epic scale, detailed architecture, cinematic lighting",
    "Anime": "anime character portrait, expressive eyes, detailed hair, clean line art, soft color palette",
    "Cinematic": "cinematic scene of a lone explorer standing in a neon-lit rainy street, film still, dramatic composition",
}
IMG2IMG_PROMPTS = {
    "Color edit": "Change only the main subject color while preserving the same shape, lighting, background, and composition",
    "Style shift": "Transform this image into a cinematic film still while preserving the subject and composition",
    "Product polish": "Make this look like a premium studio product photo, clean lighting, sharp details, same object",
    "Anime style": "Convert this image to a polished anime style while preserving the pose and composition",
}
INPAINT_PROMPTS = {
    "Replace object": "Replace the masked area with a realistic object that matches the original lighting and perspective",
    "Change clothing": "Change only the masked clothing color and fabric while preserving the person, pose, and background",
    "Remove object": "Fill the masked area naturally using the surrounding background",
    "Add detail": "Add realistic detail inside the masked area while matching the original image style",
}


def find_sd_executable():
    """Auto-detect available stable-diffusion executable."""
    candidates = [
        ("sd-cli.exe", "sd-cli.exe (recommended)"),
        ("sd.exe", "sd.exe (legacy)"),
    ]
    for exe_name, label in candidates:
        exe_path = SD_BIN_DIR / exe_name
        if exe_path.exists():
            return str(exe_path), label
    return None, None


SD_EXE, SD_EXE_LABEL = find_sd_executable()


def get_lora_list():
    """List available LoRA files in the loras directory."""
    if not LORA_DIR.exists():
        return []
    return [f.name for f in sorted(LORA_DIR.glob("*.safetensors"))]


def get_recent_outputs(limit=12):
    images = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        images.extend(OUTDIR.glob(pattern))
    return [str(p.absolute()) for p in sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]]


def apply_preset(preset_label):
    for name, w, h in RES_PRESETS:
        if name == preset_label:
            return w, h
    return gr.update(), gr.update()


def choose_preset_for_image(image_path, current_width=512, current_height=512):
    if not image_path:
        return gr.update(), gr.update(), gr.update()
    try:
        with Image.open(image_path) as image:
            source_width, source_height = image.size
    except (OSError, TypeError, ValueError):
        return gr.update(), gr.update(), gr.update()

    if source_width <= 0 or source_height <= 0:
        return gr.update(), gr.update(), gr.update()

    source_ratio = source_width / source_height
    target_area = safe_int(current_width, 512) * safe_int(current_height, 512)

    def preset_score(preset):
        _, preset_width, preset_height = preset
        ratio_error = abs((preset_width / preset_height) - source_ratio)
        area_error = abs((preset_width * preset_height) - target_area) / max(target_area, 1)
        return ratio_error, area_error

    name, width_value, height_value = min(RES_PRESETS, key=preset_score)
    return name, width_value, height_value


def sync_img2img_size(image_path, auto_size, current_width, current_height):
    if not auto_size:
        return gr.update(), gr.update(), gr.update()
    return choose_preset_for_image(image_path, current_width, current_height)


def random_seed():
    return random.randint(0, 2_147_483_647)


def reuse_last_seed():
    if LAST_SEED is None:
        return gr.update()
    return LAST_SEED


def reuse_last_img2img_seed():
    if LAST_IMG2IMG_SEED is None:
        return gr.update()
    return LAST_IMG2IMG_SEED


def reuse_last_inpaint_seed():
    if LAST_INPAINT_SEED is None:
        return gr.update()
    return LAST_INPAINT_SEED


def refresh_loras():
    return gr.update(choices=get_lora_list())


def refresh_gallery():
    return get_recent_outputs()


def short_prompt(prompt, limit=70):
    text = " ".join((prompt or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def queue_table_rows():
    with generation_lock:
        rows = []
        for index, job in enumerate(generation_jobs, start=1):
            rows.append(
                [
                    index,
                    job["mode"],
                    short_prompt(job["prompt"]),
                    job["seed"],
                    job["status"],
                ]
            )
        return rows


def set_latest_state(image=None, status=None, time_text=None, command=None):
    global latest_image, latest_status, latest_time, latest_command
    with generation_lock:
        if image is not None:
            latest_image = image
        if status is not None:
            latest_status = status
        if time_text is not None:
            latest_time = time_text
        if command is not None:
            latest_command = command


def poll_ui_state():
    with generation_lock:
        image = latest_image if latest_image else gr.update()
        status_text = latest_status
        time_text = latest_time
        command_text = latest_command
    return queue_table_rows(), image, status_text, time_text, command_text, get_recent_outputs()


def next_queued_job():
    global current_job_id, stop_requested
    with generation_lock:
        for job in generation_jobs:
            if job["status"] == "queued":
                job["status"] = "running"
                current_job_id = job["id"]
                stop_requested = False
                return job
    return None


def update_job_status(job_id, status):
    global current_job_id
    with generation_lock:
        for job in generation_jobs:
            if job["id"] == job_id:
                job["status"] = status
                break
        if current_job_id == job_id and status in {"done", "failed", "stopped"}:
            current_job_id = None


def clear_waiting_jobs():
    with generation_lock:
        generation_jobs[:] = [job for job in generation_jobs if job["status"] not in {"done", "failed", "stopped"}]
    return queue_table_rows()


def apply_example(prompt_map, selected_label):
    if selected_label in prompt_map:
        return prompt_map[selected_label]
    return gr.update()


def apply_txt2img_example(selected_label):
    return apply_example(TXT2IMG_PROMPTS, selected_label)


def apply_img2img_example(selected_label):
    return apply_example(IMG2IMG_PROMPTS, selected_label)


def apply_inpaint_example(selected_label):
    return apply_example(INPAINT_PROMPTS, selected_label)


def set_unlocked(enabled):
    return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled))


def set_img2img_enabled(enabled):
    return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled))


def set_inpaint_enabled(enabled):
    return gr.update(interactive=bool(enabled)), gr.update(interactive=bool(enabled))


def stop_gen():
    global current_proc, stop_requested
    stop_requested = True
    if current_job_id:
        update_job_status(current_job_id, "stopping")
    if current_proc and current_proc.poll() is None:
        print("Stopping generation...")
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(current_proc.pid)], capture_output=True)
        else:
            current_proc.terminate()
        return "Generation stopped by user.", queue_table_rows()
    return "No active generation to stop.", queue_table_rows()


def normalize_seed(seed_value):
    try:
        seed_int = int(seed_value)
    except (TypeError, ValueError):
        seed_int = -1
    if seed_int < 0:
        return random_seed()
    return seed_int


def seed_field_update(seed_value, run_seed):
    try:
        if int(seed_value) >= 0:
            return run_seed
    except (TypeError, ValueError):
        pass
    return gr.update()


def safe_int(value, fallback):
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def safe_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def run_generation_job(job):
    global FIRST_RUN, LAST_SEED, LAST_IMG2IMG_SEED, LAST_INPAINT_SEED, stop_requested

    generation_mode = job["mode"]
    if generation_mode == "inpaint":
        LAST_INPAINT_SEED = job["seed"]
        active_prompt = job["selective_prompt"]
        active_steps = job["selective_steps"]
        active_negative_prompt = job["selective_negative_prompt"]
        active_guidance = job["selective_guidance"]
        active_strength = job["selective_strength"]
        active_init_image, active_mask, prep_error = prepare_inpaint_images(
            job["inpaint_image"],
            job["width"],
            job["height"],
        )
        if prep_error:
            update_job_status(job["id"], "failed")
            set_latest_state(status=prep_error, time_text="Generation Time: **0s**")
            return
    elif generation_mode == "img2img":
        LAST_IMG2IMG_SEED = job["seed"]
        active_prompt = job["image_prompt"]
        active_steps = job["image_steps"]
        active_negative_prompt = job["image_negative_prompt"]
        active_guidance = job["image_guidance"]
        active_strength = job["image_strength"]
        active_init_image = job["input_image"]
        active_mask = None
    else:
        LAST_SEED = job["seed"]
        active_prompt = job["txt_prompt"]
        active_steps = job["steps"]
        active_negative_prompt = ""
        active_guidance = 3.5
        active_strength = 0.55
        active_init_image = None
        active_mask = None

    status_msg = "Generating... (first run can take longer)" if FIRST_RUN else "Generating..."
    FIRST_RUN = False
    set_latest_state(status=status_msg, time_text="Generation Time: **0s**")

    last_img = None
    last_log = ""
    last_time = "0s"
    last_command = ""
    try:
        for out_img, log, time_str, cmd_str in gen_image(
            active_prompt,
            job["width"],
            job["height"],
            active_steps,
            job["seed"],
            job["cfg_scale"],
            job["vae_path"],
            job["llm_path"],
            job["selected_loras"],
            job["lora_strength"],
            job["lora_apply_mode"],
            job["vram_mode"],
            job["clip_on_cpu"],
            job["balanced_vae_tiling"],
            job.get("cpu_threads", -1),
            job.get("enable_dit_cache", True),
            active_negative_prompt,
            active_guidance,
            generation_mode == "img2img",
            active_init_image,
            active_strength,
            active_mask,
            generation_mode,
        ):
            if out_img is not None:
                last_img = out_img
            last_log = log or ""
            last_time = time_str or "0s"
            last_command = cmd_str or ""
            set_latest_state(
                image=out_img,
                status=last_log,
                time_text=f"Generation Time: **{last_time}**",
                command=last_command,
            )
    except Exception as exc:
        update_job_status(job["id"], "failed")
        set_latest_state(status=f"Generation failed unexpectedly: {exc}")
        gc.collect()
        return

    if stop_requested or "Generation stopped" in last_log:
        update_job_status(job["id"], "stopped")
        stop_requested = False
    elif last_img is not None:
        update_job_status(job["id"], "done")
    else:
        update_job_status(job["id"], "failed")

    final_image = last_img if last_img is not None else None
    set_latest_state(
        image=final_image,
        status=last_log,
        time_text=f"Generation Time: **{last_time}**",
        command=last_command,
    )
    gc.collect()



def queue_worker():
    while True:
        job = next_queued_job()
        if not job:
            time.sleep(0.5)
            job = next_queued_job()
            if not job:
                return
        run_generation_job(job)


def ensure_generation_worker():
    global generation_worker_thread
    with worker_lock:
        if generation_worker_thread and generation_worker_thread.is_alive():
            return
        generation_worker_thread = threading.Thread(target=queue_worker, daemon=True)
        generation_worker_thread.start()


def append_lora_tags(prompt, selected_loras, lora_strength):
    final_prompt = prompt or ""
    if selected_loras:
        for lora in selected_loras:
            lora_name = Path(lora).stem
            final_prompt += f" <lora:{lora_name}:{lora_strength}>"
    return final_prompt


def low_vram_flags(vram_mode, clip_on_cpu, balanced_vae_tiling, cpu_threads=-1, enable_dit_cache=True):
    flags = []
    if cpu_threads and int(cpu_threads) > 0:
        flags.extend(["-t", str(cpu_threads)])
    if enable_dit_cache:
        flags.extend(["--cache-mode", "easycache", "--cache-option", "threshold=0.25"])
    if vram_mode == "4GB (safest)":
        flags.extend(["--offload-to-cpu", "--diffusion-fa", "--vae-tiling", "--vae-conv-direct"])
        if clip_on_cpu:
            flags.append("--clip-on-cpu")
    elif vram_mode == "6-8GB (balanced)":
        flags.extend(["--offload-to-cpu", "--diffusion-fa"])
        if balanced_vae_tiling:
            flags.append("--vae-tiling")
    elif vram_mode == "10GB+ (fastest)":
        flags.append("--diffusion-fa")
    return flags



def upscale_image(image_path, scale_factor=2.0):
    """Upscale image using Lanczos resampling with adaptive unsharp mask sharpening."""
    if not image_path or not os.path.exists(image_path):
        return None, "No image selected for upscaling."
    try:
        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            w, h = img.size
            new_w, new_h = int(w * float(scale_factor)), int(h * float(scale_factor))

            upscaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            upscaled = upscaled.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

            uid = uuid.uuid4().hex[:8]
            out_file = str((OUTDIR / f"upscaled_{new_w}x{new_h}_{uid}.png").absolute())
            upscaled.save(out_file, quality=95)

            gc.collect()
            return out_file, f"Successfully upscaled image to {new_w}x{new_h}!"
    except Exception as exc:
        return None, f"Error upscaling image: {exc}"



def prepare_init_image(init_image_path, width, height):
    if not init_image_path:
        return None, None

    src = Path(init_image_path)
    if not src.exists():
        return None, f"Img2img input image was not found: {src}"

    try:
        image = Image.open(src)
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        dest = TEMP_INPUT_DIR / f"init_{uuid.uuid4().hex[:8]}.png"
        image.save(dest)
        return str(dest.absolute()), None
    except Exception as exc:
        return None, f"Could not prepare img2img input image: {exc}"


def get_editor_background(editor_value):
    if not editor_value:
        return None
    if isinstance(editor_value, dict):
        return editor_value.get("background") or editor_value.get("composite")
    return editor_value


def prepare_inpaint_images(editor_value, width, height):
    if not editor_value:
        return None, None, "Upload an image and paint a mask before using inpainting."

    background = get_editor_background(editor_value)
    if background is None:
        return None, None, "Inpaint source image is missing."

    try:
        image = ImageOps.exif_transpose(background).convert("RGB")
        layers = editor_value.get("layers", []) if isinstance(editor_value, dict) else []
        mask = Image.new("L", image.size, 0)
        for layer in layers:
            layer_img = ImageOps.exif_transpose(layer).convert("RGBA")
            alpha = layer_img.getchannel("A")
            mask = Image.composite(Image.new("L", image.size, 255), mask, alpha)

        if not mask.getbbox():
            return None, None, "Paint over the area you want to edit before generating."

        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        uid = uuid.uuid4().hex[:8]
        init_dest = TEMP_INPUT_DIR / f"inpaint_init_{uid}.png"
        mask_dest = TEMP_INPUT_DIR / f"inpaint_mask_{uid}.png"
        image.save(init_dest)
        mask.save(mask_dest)
        return str(init_dest.absolute()), str(mask_dest.absolute()), None
    except Exception as exc:
        return None, None, f"Could not prepare inpaint image/mask: {exc}"


def format_command(cmd):
    return subprocess.list2cmdline([str(part) for part in cmd])


def write_metadata(out_file, metadata):
    meta_path = Path(out_file).with_suffix(Path(out_file).suffix + ".txt")
    lines = [
        "Z-Image Turbo generation metadata",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        lines.append(f"{key}: {value}")
    try:
        meta_path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        print(f"Could not write metadata file: {exc}")


def ensure_model_file(model_key):
    import json
    import requests
    from tqdm import tqdm

    registry_path = ROOT / "config" / "model_registry.json"
    if not registry_path.exists():
        return None
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
        model_info = data.get("models", {}).get(model_key)
        if not model_info:
            return None

        m_type = model_info.get("type", "")
        if m_type == "vae":
            dest_dir = ROOT / "models" / "vae"
        elif m_type in ("llm", "t5"):
            dest_dir = ROOT / "models" / "llm"
        else:
            dest_dir = ROOT / "models" / "zimage"

        dest_file = dest_dir / model_info["filename"]
        if dest_file.exists() and dest_file.stat().st_size > 50 * 1024 * 1024:
            return str(dest_file.absolute())

        url = model_info.get("url")
        if not url:
            return None

        print(f"Downloading missing model {model_info['display_name']} from HuggingFace...")
        dest_dir.mkdir(parents=True, exist_ok=True)

        response = requests.get(url, stream=True, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        temp_file = dest_file.with_suffix(".tmp")
        with open(temp_file, "wb") as f, tqdm(
            desc=model_info["filename"],
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data_chunk in response.iter_content(chunk_size=1024 * 1024):
                if data_chunk:
                    size = f.write(data_chunk)
                    bar.update(size)
        if temp_file.exists():
            if dest_file.exists():
                dest_file.unlink()
            temp_file.rename(dest_file)
        return str(dest_file.absolute())
    except Exception as exc:
        print(f"Auto-download model failed: {exc}")
        return None


def gen_image(
    prompt,
    width,
    height,
    steps,
    seed,
    cfg_scale,
    vae_path,
    llm_path,
    selected_loras,
    lora_strength,
    lora_apply_mode,
    vram_mode,
    clip_on_cpu,
    balanced_vae_tiling,
    cpu_threads,
    enable_dit_cache,
    negative_prompt,
    guidance,
    img2img_enabled,
    init_image_path,
    img2img_strength,
    mask_path=None,
    generation_mode="txt2img",
):
    global current_proc

    if SD_EXE is None:
        yield None, "Error: No stable-diffusion executable found.", "", ""
        return

    uses_input_image = img2img_enabled or generation_mode == "inpaint"
    if uses_input_image and not init_image_path:
        yield None, f"{generation_mode} is enabled, but no input image was provided.", "0s", ""
        return

    init_file = None
    init_error = None
    if generation_mode == "inpaint":
        init_file = init_image_path
    elif img2img_enabled:
        init_file, init_error = prepare_init_image(init_image_path, width, height)
    if init_error:
        yield None, init_error, "0s", ""
        return

    uid = uuid.uuid4().hex[:8]
    out_file = str((OUTDIR / f"out_{uid}.png").absolute())
    diffusion_model = MODEL_PATH

    final_prompt = append_lora_tags(prompt, selected_loras, lora_strength)

    cmd = [
            SD_EXE,
            "--diffusion-model",
            diffusion_model,
            "--vae",
            vae_path,
            "--llm",
            llm_path,
            "--lora-model-dir",
            str(LORA_DIR),
            "--lora-apply-mode",
            lora_apply_mode,
            "-p",
            final_prompt,
            "--guidance",
            str(guidance),
            "--cfg-scale",
            str(cfg_scale),
            "--steps",
            str(steps),
            "-H",
            str(height),
            "-W",
            str(width),
            "-o",
            out_file,
            "--seed",
            str(seed),
            "--rng",
            "cuda",
        ]

    if negative_prompt:
        cmd.extend(["--negative-prompt", negative_prompt])
    cmd.extend(low_vram_flags(vram_mode, clip_on_cpu, balanced_vae_tiling, cpu_threads, enable_dit_cache))

    if uses_input_image:
        cmd.extend(["--init-img", init_file, "--strength", str(img2img_strength)])
    if generation_mode == "inpaint":
        cmd.extend(["--mask", mask_path])

    cmd_str = format_command(cmd)
    yield None, f"Starting generation...\nCommand: {cmd_str}", "0s", cmd_str

    t_start = time.perf_counter()
    try:
        current_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except OSError as exc:
        yield None, f"Could not start stable-diffusion.cpp backend: {exc}", "0s", cmd_str
        return

    full_log = ""
    try:
        for line in current_proc.stdout:
            print(line, end="")
            full_log += line
            elapsed = int(time.perf_counter() - t_start)
            yield None, full_log.strip(), f"{elapsed}s", cmd_str
    except Exception as exc:
        yield None, f"Error during logging: {exc}", "0s", cmd_str

    current_proc.wait()
    total_time = f"{time.perf_counter() - t_start:.1f}s"

    if current_proc.returncode != 0:
        if current_proc.returncode in [-1, 1, 3221225786, 15]:
            yield None, f"Generation stopped.\n\n{full_log.strip()}", total_time, cmd_str
        else:
            yield (
                None,
                f"sd.exe exited with code {current_proc.returncode}\n\n{full_log.strip()}",
                total_time,
                cmd_str,
            )
        gc.collect()
        return

    if not os.path.exists(out_file):
        imgs = sorted(OUTDIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not imgs:
            yield None, f"No image was produced.\n\n{full_log.strip()}", total_time, cmd_str
            gc.collect()
            return
        out_file = str(imgs[0].absolute())

    write_metadata(
        out_file,
        {
            "mode": generation_mode,
            "prompt": prompt,
            "final_prompt": final_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "guidance": guidance,
            "negative_prompt": negative_prompt,
            "vram_mode": vram_mode,
            "lora_files": ", ".join(selected_loras or []),
            "lora_strength": lora_strength,
            "lora_apply_mode": lora_apply_mode,
            "init_image": init_file,
            "mask": mask_path,
            "strength": img2img_strength if uses_input_image else None,
            "command": cmd_str,
            "generation_time": total_time,
            "log": full_log.strip(),
        },
    )
    gc.collect()
    yield out_file, full_log.strip(), total_time, cmd_str



CUSTOM_CSS = """
.gradio-container { max-width: 100% !important; padding: 4px !important; }
.compact-row { gap: 6px !important; margin-bottom: 2px !important; }
.compact-group { padding: 6px !important; margin-bottom: 4px !important; border-radius: 6px; }
button { font-weight: 600 !important; }
"""

with gr.Blocks(css=CUSTOM_CSS, title="Z-Image Turbo Dashboard") as demo:
    gr.Markdown("# ⚡ Z-Image Turbo — Extreme Performance Dashboard")
    queue_refresh_timer = gr.Timer(1.0)

    with gr.Row():
        # LEFT COLUMN (Controls & Inputs)
        with gr.Column(scale=5):
            gen_mode = gr.Radio(
                ["Text-to-Image", "Image-to-Image", "Inpaint"],
                value="Text-to-Image",
                label="Modo de Geração",
                interactive=True,
            )

            prompt = gr.Textbox(
                label="Prompt",
                value="A large orange octopus on an ocean floor, cinematic, 8k",
                lines=2,
                placeholder="Descreva a imagem que deseja gerar...",
            )

            with gr.Row():
                example_dropdown = gr.Dropdown(
                    list(TXT2IMG_PROMPTS.keys()),
                    value="Portrait",
                    label="Presets de Estilo",
                    scale=3,
                )
                apply_example_btn = gr.Button("Usar Preset", variant="secondary", scale=1)

            # Conditional Image Inputs Container
            with gr.Group(visible=False) as img2img_group:
                init_image = gr.Image(label="Imagem de Origem (Img2Img)", type="filepath", interactive=True)
                img2img_strength = gr.Slider(0.1, 1.0, value=0.55, step=0.05, label="Força da Alteração (Strength)")

            with gr.Group(visible=False) as inpaint_group:
                inpaint_editor = gr.ImageEditor(
                    label="Editor de Máscara (Pinte a área a ser editada)",
                    type="pil",
                    image_mode="RGBA",
                    brush=Brush(default_size=32, colors=["#ffffff"], default_color="#ffffff", color_mode="fixed"),
                    eraser=Eraser(default_size=32),
                    layers=True,
                    interactive=True,
                    height=320,
                )
                inpaint_strength = gr.Slider(0.1, 1.0, value=0.75, step=0.05, label="Força do Inpaint")

            # Generation Settings Row
            with gr.Row():
                preset = gr.Dropdown([n for n, _, _ in RES_PRESETS], value="1:1 (512x512)", label="Resolução")
                width = gr.Dropdown(SIZE_OPTIONS, value=512, label="Largura")
                height = gr.Dropdown(SIZE_OPTIONS, value=512, label="Altura")
                steps = gr.Slider(1, 30, value=4, step=1, label="Passos (Steps)")
                cfg_scale = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="CFG Scale")

            with gr.Row():
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatório)")
                batch_count = gr.Slider(1, 16, value=1, step=1, label="Quantidade de Imagens")
                random_seed_btn = gr.Button("🎲 Nova Seed", variant="secondary", size="sm")
                reuse_seed_btn = gr.Button("🔄 Reutilizar Seed", variant="secondary", size="sm")

            # Accordion: Extreme Performance & Low VRAM
            with gr.Accordion("⚡ Otimizações Extremas & Low VRAM", open=False):
                with gr.Row():
                    vram_mode = gr.Radio(VRAM_PRESETS, value="4GB (safest)", label="Preset de VRAM")
                    enable_dit_cache = gr.Checkbox(value=True, label="Ativar DiT Cache (easycache +35% FPS)")
                with gr.Row():
                    clip_on_cpu = gr.Checkbox(value=False, label="Text Encoder na RAM (4GB Extra)")
                    balanced_vae_tiling = gr.Checkbox(value=True, label="VAE Tiling")
                    cpu_threads = gr.Slider(1, os.cpu_count() or 16, value=os.cpu_count() or 4, step=1, label="Threads da CPU")

            # Accordion: LoRA Models
            with gr.Accordion("🎨 Modelos LoRA", open=False):
                with gr.Row():
                    lora_list = gr.CheckboxGroup(choices=get_lora_list(), label="LoRAs Selecionados")
                    refresh_loras_btn = gr.Button("Atualizar LoRAs", variant="secondary", size="sm")
                with gr.Row():
                    lora_strength = gr.Slider(0.0, 2.0, value=1.0, step=0.1, label="Força do LoRA")
                    lora_apply_mode = gr.Dropdown(LORA_APPLY_MODES, value="auto", label="Modo de Aplicação")

            # Accordion: Advanced Options
            with gr.Accordion("⚙️ Caminhos Avançados", open=False):
                unlock = gr.Checkbox(value=False, label="Editar caminhos dos modelos")
                vae_path = gr.Textbox(label="Caminho VAE", value=DEFAULT_VAE_PATH, interactive=False)
                llm_path = gr.Textbox(label="Caminho LLM (Qwen)", value=DEFAULT_LLM_PATH, interactive=False)

            # Action Buttons
            with gr.Row():
                btn = gr.Button("⚡ GERAR IMAGEM", variant="primary", scale=3)
                stop_btn = gr.Button("🛑 PARAR", variant="stop", scale=1)

        # RIGHT COLUMN (Outputs & Recent Gallery Dashboard)
        with gr.Column(scale=4):
            img = gr.Image(label="Resultado da Geração", interactive=False, type="filepath", height=340)
            with gr.Row():
                upscale_2x_btn = gr.Button("🔍 Upscale 2x HD", variant="secondary")
                refresh_gallery_btn = gr.Button("🖼️ Atualizar Galeria", variant="secondary")
            timer_display = gr.Markdown("Tempo de Geração: **0s**")

            with gr.Accordion("🖼️ Galeria de Arquivos Recentes", open=True):
                gallery = gr.Gallery(value=get_recent_outputs(), columns=4, height=180)

            with gr.Accordion("📋 Status & Fila de Execução", open=False):
                command_box = gr.Textbox(label="Último Comando Executado", interactive=False, lines=2)
                queue_table = gr.Dataframe(
                    headers=["#", "Modo", "Prompt", "Seed", "Status"],
                    value=queue_table_rows(),
                    datatype=["number", "str", "str", "number", "str"],
                    interactive=False,
                    label="Fila de Geração",
                    row_count=(3, "dynamic"),
                    column_count=(5, "fixed"),
                )
                clear_queue_btn = gr.Button("Limpar Fila Concluída", variant="secondary", size="sm")
                status = gr.Textbox(label="Logs de Status", interactive=False, lines=6)

    # Dynamic visibility switching logic
    def toggle_mode_views(selected_mode):
        show_img2img = selected_mode == "Image-to-Image"
        show_inpaint = selected_mode == "Inpaint"
        return (
            gr.update(visible=show_img2img),
            gr.update(visible=show_inpaint),
        )

    gen_mode.change(
        toggle_mode_views,
        inputs=[gen_mode],
        outputs=[img2img_group, inpaint_group],
    )

    def apply_selected_example(style_key):
        return TXT2IMG_PROMPTS.get(style_key, "")

    preset.change(apply_preset, inputs=[preset], outputs=[width, height])
    apply_example_btn.click(apply_selected_example, inputs=[example_dropdown], outputs=[prompt])
    random_seed_btn.click(random_seed, outputs=[seed])
    reuse_seed_btn.click(reuse_last_seed, outputs=[seed])
    refresh_loras_btn.click(refresh_loras, outputs=[lora_list])
    refresh_gallery_btn.click(refresh_gallery, outputs=[gallery])
    upscale_2x_btn.click(upscale_image, inputs=[img], outputs=[img, status])
    unlock.change(set_unlocked, inputs=[unlock], outputs=[vae_path, llm_path])

    state_outputs = [queue_table, img, status, timer_display, command_box, gallery]
    demo.load(poll_ui_state, outputs=state_outputs, queue=False, show_progress="hidden")
    queue_refresh_timer.tick(poll_ui_state, outputs=state_outputs, queue=False, show_progress="hidden")

    def submit_unified_job(
        selected_mode,
        user_prompt,
        user_width,
        user_height,
        user_steps,
        user_cfg,
        user_seed,
        batch_cnt,
        vram_preset,
        use_dit_cache,
        clip_cpu,
        vae_tiling,
        threads_count,
        selected_lora_list,
        lora_str,
        lora_mode,
        vae_p,
        llm_p,
        input_img_path,
        img2img_str,
        inpaint_editor_val,
        inpaint_str,
    ):
        if selected_mode == "Inpaint":
            mode_code = "inpaint"
        elif selected_mode == "Image-to-Image":
            mode_code = "img2img"
        else:
            mode_code = "txt2img"

        num_images = max(1, safe_int(batch_cnt, 1))

        prep_init = None
        prep_mask = None

        if mode_code == "img2img":
            if not input_img_path:
                return queue_table_rows(), "Erro: Selecione uma imagem para o modo Img2Img."
            prep_init, err = prepare_init_image(input_img_path, safe_int(user_width, 512), safe_int(user_height, 512))
            if err:
                return queue_table_rows(), err
        elif mode_code == "inpaint":
            if not inpaint_editor_val:
                return queue_table_rows(), "Erro: Pinte uma área na imagem para o modo Inpaint."
            prep_init, prep_mask, err = prepare_inpaint_images(inpaint_editor_val, safe_int(user_width, 512), safe_int(user_height, 512))
            if err:
                return queue_table_rows(), err

        new_jobs = []
        try:
            base_seed = int(user_seed)
        except (TypeError, ValueError):
            base_seed = -1

        for i in range(num_images):
            run_seed = random_seed() if base_seed < 0 else base_seed + i
            job = {
                "id": uuid.uuid4().hex,
                "mode": mode_code,
                "status": "queued",
                "prompt": user_prompt,
                "txt_prompt": user_prompt,
                "image_prompt": user_prompt,
                "selective_prompt": user_prompt,
                "width": safe_int(user_width, 512),
                "height": safe_int(user_height, 512),
                "steps": safe_int(user_steps, 4),
                "txt_seed": user_seed,
                "image_seed": user_seed,
                "selective_seed": user_seed,
                "seed": run_seed,
                "cfg_scale": safe_float(user_cfg, 1.0),
                "vae_path": vae_p,
                "llm_path": llm_p,
                "selected_loras": list(selected_lora_list or []),
                "lora_strength": safe_float(lora_str, 1.0),
                "lora_apply_mode": lora_mode,
                "vram_mode": vram_preset,
                "clip_on_cpu": bool(clip_cpu),
                "balanced_vae_tiling": bool(vae_tiling),
                "cpu_threads": safe_int(threads_count, -1),
                "enable_dit_cache": bool(use_dit_cache),
                "input_image": prep_init,
                "image_negative_prompt": "",
                "image_steps": safe_int(user_steps, 4),
                "image_guidance": 3.5,
                "image_strength": safe_float(img2img_str, 0.55) if mode_code == "img2img" else safe_float(inpaint_str, 0.75),
                "inpaint_image": prep_init,
                "inpaint_mask": prep_mask,
                "selective_negative_prompt": "",
                "selective_steps": safe_int(user_steps, 4),
                "selective_guidance": 4.0,
                "selective_strength": safe_float(inpaint_str, 0.75),
            }
            new_jobs.append(job)

        with generation_lock:
            generation_jobs.extend(new_jobs)
            queued_count = sum(1 for item in generation_jobs if item["status"] == "queued")
            running_count = sum(1 for item in generation_jobs if item["status"] == "running")

        msg = f"Adicionado(s) {len(new_jobs)} job(s) à fila ({mode_code}). Executando: {running_count} | Aguardando: {queued_count}"
        ensure_generation_worker()
        return queue_table_rows(), msg

    btn.click(
        submit_unified_job,
        inputs=[
            gen_mode,
            prompt,
            width,
            height,
            steps,
            cfg_scale,
            seed,
            batch_count,
            vram_mode,
            enable_dit_cache,
            clip_on_cpu,
            balanced_vae_tiling,
            cpu_threads,
            lora_list,
            lora_strength,
            lora_apply_mode,
            vae_path,
            llm_path,
            init_image,
            img2img_strength,
            inpaint_editor,
            inpaint_strength,
        ],
        outputs=[queue_table, status],
        queue=False,
        trigger_mode="multiple",
        show_progress="hidden",
    )
    clear_queue_btn.click(clear_waiting_jobs, outputs=[queue_table], queue=False, show_progress="hidden")
    stop_btn.click(stop_gen, outputs=[status, queue_table], queue=False, show_progress="hidden")

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="127.0.0.1",
        server_port=9000,
        share=False,
        quiet=os.environ.get("ZIMAGE_QUIET_LAUNCH") == "1",
    )


