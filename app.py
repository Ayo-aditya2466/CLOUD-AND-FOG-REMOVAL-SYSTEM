import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from models.gen.SPANet import Generator
from utils import heatmap


ROOT = Path(__file__).resolve().parent
MODELS = {
    "RICE2 (cloud removal)": ROOT / "pretrained_models" / "RICE2" / "gen_model_epoch_200.pth",
    "RICE1 (fog removal)": ROOT / "pretrained_models" / "RICE1" / "gen_model_epoch_200.pth",
}


@st.cache_resource
def load_model(model_path, device_name):
    device = torch.device(device_name)
    gpu_ids = [0] if device.type == "cuda" else []
    model = Generator(gpu_ids=gpu_ids)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def restore_image(image_rgb, model, device):
    image_bgr = cv2.cvtColor(np.asarray(image_rgb), cv2.COLOR_RGB2BGR)
    image = image_bgr.astype(np.float32) / 255.0
    image = image.transpose(2, 0, 1)[None]
    tensor = torch.from_numpy(image).to(device)

    with torch.no_grad():
        attention, restored = model(tensor)

    restored_bgr = np.clip(restored[0].cpu().numpy()[:3], 0, 1)
    restored_bgr = (restored_bgr.transpose(1, 2, 0) * 255).astype(np.uint8)
    attention_map = (attention[0].cpu().numpy() * 255).astype(np.uint8)
    attention_bgr = heatmap(attention_map)[0].transpose(1, 2, 0)

    return (
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(attention_bgr, cv2.COLOR_BGR2RGB),
    )


def get_sample_image():
    sample_path = ROOT / "readme_images" / "test_0000.png"
    with Image.open(sample_path) as sample:
        width, height = sample.size
        return sample.convert("RGB").crop((0, 0, width // 4, height))


st.set_page_config(page_title="Cloud & Fog Removal", page_icon="🛰️", layout="wide")
st.title("🛰️ Satellite Cloud & Fog Removal")
st.write("Restore a satellite image with a pretrained model, then download the result.")
st.info(
    "No image handy? Choose the included sample below. The bundled dataset folders "
    "are placeholders, but the pretrained models and demo sample are included."
)

with st.sidebar:
    st.header("Settings")
    model_label = st.selectbox("Pretrained model", list(MODELS))
    use_gpu = st.checkbox(
        "Use CUDA GPU",
        value=False,
        disabled=not torch.cuda.is_available(),
    )
    st.caption("CPU inference works without a GPU.")

image_source = st.radio("Image source", ["Use the included sample", "Upload an image"], horizontal=True)
if image_source == "Upload an image":
    uploaded_file = st.file_uploader("Choose a satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    input_image = Image.open(uploaded_file).convert("RGB") if uploaded_file else None
    image_name = uploaded_file.name if uploaded_file else "restored.png"
else:
    input_image = get_sample_image()
    image_name = "sample.png"

if input_image is not None:
    st.image(input_image, caption="Input image", width=512)
    if st.button("Remove clouds / fog", type="primary"):
        model_path = MODELS[model_label]
        if not model_path.is_file():
            st.error("The selected pretrained model was not found: {}".format(model_path))
        else:
            device_name = "cuda:0" if use_gpu else "cpu"
            with st.spinner("Loading model and restoring image..."):
                model = load_model(str(model_path), device_name)
                input_result, restored_result, attention_result = restore_image(
                    input_image, model, torch.device(device_name)
                )
            st.session_state["restoration_result"] = (
                input_result,
                restored_result,
                attention_result,
            )
            st.session_state["restoration_name"] = image_name

if "restoration_result" in st.session_state:
    input_result, restored_result, attention_result = st.session_state["restoration_result"]
    st.subheader("Restoration result")
    columns = st.columns(3)
    for column, image, caption in zip(
        columns,
        (input_result, restored_result, attention_result),
        ("Input", "Restored", "Attention map"),
    ):
        column.image(image, caption=caption, use_container_width=True)

    output = io.BytesIO()
    Image.fromarray(restored_result).save(output, format="PNG")
    st.download_button(
        "Download restored image",
        data=output.getvalue(),
        file_name=Path(st.session_state["restoration_name"]).stem + "_restored.png",
        mime="image/png",
    )
