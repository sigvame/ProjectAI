import torch
import time
import os
from diffusers import StableDiffusionPipeline
from PIL import Image

MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "sd_lab_results"
PROMPTS = {
    "steps_ukr": "Гірський пейзаж з водоспадом на заході сонця, деталізований",
    "steps_eng": "Mountain landscape with waterfall at sunset, highly detailed",
    "guidance_ukr": "Кіт у космічному скафандрі на Місяці, реалістичний",
    "guidance_eng": "Cat in space suit on the Moon, photorealistic",
    "size_ukr": "Старовинний замок у тумані, акварель",
    "size_eng": "Ancient castle in the fog, watercolor"
}


def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype
        )
        pipe = pipe.to(device)
    except Exception as e:
        print(f"Помилка завантаження: {e}")
        return None

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return pipe


def generate_and_time(pipe, prompt: str, steps: int, guidance: float, width: int, height: int, filename: str) -> tuple[
    float, str]:
    if pipe is None:
        return 0.0, ""

    start_time = time.time()

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height
    )
    image = result.images[0]

    end_time = time.time()
    generation_time = end_time - start_time

    save_path = os.path.join(OUTPUT_DIR, filename)
    image.save(save_path)

    return generation_time, save_path


# def download_model():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.float16 if device == "cuda" else torch.float32
#     StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)

if __name__ == "__main__":

    # download_model()

    pipe = load_model()

    if pipe is None:
        exit()

    steps_values = [10, 20, 30, 50]
    fixed_guidance = 7.5
    fixed_width = 512
    fixed_height = 512

    print("\n--- Вплив кроків")

    for steps in steps_values:
        time_ukr, path_ukr = generate_and_time(
            pipe, PROMPTS["steps_ukr"], steps, fixed_guidance, fixed_width, fixed_height, f"T1_S{steps}_ukr.png"
        )
        time_eng, path_eng = generate_and_time(
            pipe, PROMPTS["steps_eng"], steps, fixed_guidance, fixed_width, fixed_height, f"T1_S{steps}_eng.png"
        )
        print(f"Кроки: {steps} | Час (УКР): {time_ukr:.2f}с | Час (АНГЛ): {time_eng:.2f}с")

    guidance_values = [3.0, 7.5, 12.0, 15.0]
    fixed_steps = 30

    print("\n--- Вплив guidance_scale ---")

    for guidance in guidance_values:
        generate_and_time(
            pipe, PROMPTS["guidance_ukr"], fixed_steps, guidance, fixed_width, fixed_height,
            f"T2_G{guidance:.1f}_ukr.png"
        )
        generate_and_time(
            pipe, PROMPTS["guidance_eng"], fixed_steps, guidance, fixed_width, fixed_height,
            f"T2_G{guidance:.1f}_eng.png"
        )
        print(f"Guidance: {guidance:.1f} | Зображення збережено")

    size_values = [(256, 256), (512, 512), (768, 768)]

    print("\n--- Вплив розміру ---")

    for w, h in size_values:
        time_ukr, _ = generate_and_time(
            pipe, PROMPTS["size_ukr"], fixed_steps, fixed_guidance, w, h, f"T3_Size{w}_ukr.png"
        )
        time_eng, _ = generate_and_time(
            pipe, PROMPTS["size_eng"], fixed_steps, fixed_guidance, w, h, f"T3_Size{w}_eng.png"
        )
        print(f"Розмір: {w}x{h} | Час (УКР): {time_ukr:.2f}с | Час (АНГЛ): {time_eng:.2f}с")

