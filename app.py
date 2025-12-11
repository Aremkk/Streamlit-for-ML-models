import streamlit as st
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DiffusionPipeline, EulerAncestralDiscreteScheduler

st.set_page_config(page_title="AI Generator: Text to 3D View", layout="wide")
st.title("🌌 Генератор: Текст -> Изображение -> Ракурсы")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    st.error("⚠️ Внимание! NVIDIA GPU не обнаружен. Этот код очень медленно работает или упадет на CPU.")



@st.cache_resource
def load_models():
    st.info("Загрузка моделей в память видеокарты... Это может занять пару минут.")

    # 1. Загрузка Stable Diffusion v1.5
    pipe_sd = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)

    # 2. Загрузка Zero123++
    pipe_zero = DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16
    )
    pipe_zero.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe_zero.scheduler.config, timestep_spacing='trailing'
    )
    pipe_zero.to(device)

    return pipe_sd, pipe_zero


try:
    sd_pipeline, zero_pipeline = load_models()
    st.success("Модели успешно загружены!")
except Exception as e:
    st.error(f"Ошибка при загрузке моделей: {e}")
    st.stop()


with st.container():
    st.header("Шаг 1: Генерация основного изображения")
    prompt = st.text_input(
        "Введите описание:",
        "a dark empty house, dreamcore art, cozy lighting"
    )

    generate_btn = st.button("Сгенерировать изображение", type="primary")

    if generate_btn and prompt:
        with st.spinner("Генерация Stable Diffusion..."):
            image = sd_pipeline(prompt).images[0]

            st.session_state['base_image'] = image
            st.success("Изображение готово!")

if 'base_image' in st.session_state:
    st.image(st.session_state['base_image'], caption="Базовое изображение", width=512)

    st.divider()

    st.header("Шаг 2: Генерация ракурсов (Zero123++)")
    st.write("Генерация видов с азимутов: 30°, 90°, 150°")

    if st.button("Сгенерировать ракурсы"):
        with st.spinner("Генерация ракурсов (это может занять время)..."):
            source_img = st.session_state['base_image'].convert("RGB")

            generated_views = []
            angles = [30, 90, 150]
            for azim in angles:
                out = zero_pipeline(source_img, azimuth=azim).images[0]
                generated_views.append((azim, out))

            cols = st.columns(len(angles))
            for idx, (azim, view_img) in enumerate(generated_views):
                with cols[idx]:
                    st.image(view_img, caption=f"Азимут: {azim}°")