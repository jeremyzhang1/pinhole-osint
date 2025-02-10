from pathlib import Path
import spaces
import gradio as gr
import imageio
import torch
from PIL import Image
from omegaconf import OmegaConf
from algorithms.dfot import DFoTVideoPose
from utils.ckpt_utils import download_pretrained
from datasets.video.utils.io import read_video
from datasets.video import RealEstate10KAdvancedVideoDataset
from export import export_to_video

DATASET_DIR = Path("data/real-estate-10k-tiny")
LONG_LENGTH = 20 # seconds

metadata = torch.load(DATASET_DIR / "metadata" / "test.pt", weights_only=False)
video_list = [
    read_video(path).permute(0, 3, 1, 2) / 255.0 for path in metadata["video_paths"]
]
first_frame_list = [
    (video[0] * 255).permute(1, 2, 0).numpy().clip(0, 255).astype("uint8")
    for video in video_list
]
poses_list = [
    torch.cat(
        [
            poses[:, :4],
            poses[:, 6:],
        ],
        dim=-1,
    ).to(torch.float32)
    for poses in (
        torch.load(DATASET_DIR / "test_poses" / f"{path.stem}.pt")
        for path in metadata["video_paths"]
    )
]

# pylint: disable-next=no-value-for-parameter
dfot = DFoTVideoPose.load_from_checkpoint(
    checkpoint_path=download_pretrained("pretrained:DFoT_RE10K.ckpt"),
    cfg=OmegaConf.load("config.yaml"),
).eval()
dfot.to("cuda")

def prepare_long_gt_video(idx: int):
    video = video_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, LONG_LENGTH * 10, dtype=torch.long)
    return export_to_video(video[indices], fps=10)

@spaces.GPU(duration=120)
@torch.no_grad()
def single_image_to_long_video(idx: int, guidance_scale: float, fps: int, progress=gr.Progress(track_tqdm=True)):
    video = video_list[idx]
    poses = poses_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, LONG_LENGTH * fps, dtype=torch.long)
    xs = video[indices].unsqueeze(0).to("cuda")
    conditions = poses[indices].unsqueeze(0).to("cuda")
    dfot.cfg.tasks.prediction.history_guidance.guidance_scale = guidance_scale
    dfot.cfg.tasks.prediction.keyframe_density = 0.6 / fps
    # dfot.cfg.tasks.interpolation.history_guidance.guidance_scale = guidance_scale
    gen_video = dfot._unnormalize_x(
        dfot._predict_videos(
            dfot._normalize_x(xs),
            conditions,
        )
    )
    return export_to_video(gen_video[0].detach().cpu(), fps=fps)


# Create the Gradio Blocks
with gr.Blocks(theme=gr.themes.Base(primary_hue="teal")) as demo:
    gr.HTML(
        """
    <style>
    [data-tab-id="task-1"], [data-tab-id="task-2"], [data-tab-id="task-3"] {
        font-size: 16px !important;
        font-weight: bold;
    }
    </style>
    """
    )

    gr.Markdown("# Diffusion Forcing Transformer and History Guidance")
    gr.Markdown(
        "### Official Interactive Demo for [_History-guided Video Diffusion_](todo)"
    )
    with gr.Row():
        gr.Button(value="🌐 Website", link="todo")
        gr.Button(value="📄 Paper", link="https://boyuan.space/history-guidance")
        gr.Button(
            value="💻 Code",
            link="https://github.com/kwsong0113/diffusion-forcing-transformer",
        )
        gr.Button(
            value="🤗 Pretrained Models", link="https://huggingface.co/kiwhansong/DFoT"
        )

    with gr.Tab("Single Image → Long Video", id="task-1"):
        gr.Markdown(
            """
            ## Demo 2: Single Image → Long Video
            > #### **TL;DR:** _Diffusion Forcing Transformer, with History Guidance, can stably generate long videos, via sliding window rollouts and interpolation._
        """
        )

        stage = gr.State(value="Selection")
        selected_index = gr.State(value=None)

        @gr.render(inputs=[stage, selected_index])
        def render_stage(s, idx):
            match s:
                case "Selection":
                    image_gallery = gr.Gallery(
                        value=first_frame_list,
                        label="Select an image to animate",
                        columns=[8],
                        selected_index=idx,
                    )

                    @image_gallery.select(inputs=None, outputs=selected_index)
                    def update_selection(selection: gr.SelectData):
                        return selection.index

                    select_button = gr.Button("Select")

                    @select_button.click(inputs=selected_index, outputs=stage)
                    def move_to_generation(idx: int):
                        if idx is None:
                            gr.Warning("Image not selected!")
                            return "Selection"
                        else:
                            return "Generation"

                case "Generation":
                    with gr.Row():
                        gr.Image(value=first_frame_list[idx], label="Input Image")
                        # gr.Video(value=metadata["video_paths"][idx], label="Ground Truth Video")
                        gr.Video(value=prepare_long_gt_video(idx), label="Ground Truth Video")
                        video = gr.Video(label="Generated Video")

                        with gr.Column():
                            guidance_scale = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=4,
                                step=0.5,
                                label="History Guidance Scale",
                                info="Without history guidance: 1.0; Recommended: 4.0",
                                interactive=True,
                            )
                            fps = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=4,
                                step=1,
                                label="FPS",
                                info=f"A {LONG_LENGTH}-second video will be generated at this FPS; Decrease for faster generation; Increase for a smoother video",
                                interactive=True,
                            )
                            generate_button = gr.Button("Generate Video").click(
                                fn=single_image_to_long_video,
                                inputs=[selected_index, guidance_scale, fps],
                                outputs=video,
                            )
                    # def generate_video(idx: int):
                    #     gr.Video(value=single_image_to_long_video(idx))

        # Function to update the state with the selected index

        # def show_warning(selection: gr.SelectData):
        #     gr.Warning(f"Your choice is #{selection.index}, with image: {selection.value['image']['path']}!")

        # # image_gallery.select(fn=show_warning, inputs=None)

        # # Show the generate button only if an image is selected
        # selected_index.change(fn=lambda idx: idx is not None, inputs=selected_index, outputs=generate_button)

    with gr.Tab("Any Images → Video", id="task-2"):
        gr.Markdown(
            """
            ## Demo 1: Any Images → Video
            > #### **TL;DR:** _Diffusion Forcing Transformer is a flexible model that can generate videos given variable number of context frames._
        """
        )
        input_text_1 = gr.Textbox(
            lines=2, placeholder="Enter text for Video Model 1..."
        )
        output_video_1 = gr.Video()
        generate_button_1 = gr.Button("Generate Video")

    with gr.Tab("Single Image → Extremely Long Video", id="task-3"):
        gr.Markdown(
            """
            ## Demo 3: Single Image → Extremely Long Video
            > #### **TL;DR:** _Diffusion Forcing Transformer is a flexible model that can generate videos given **variable number of context frames**._
        """
        )
        input_text_2 = gr.Textbox(
            lines=2, placeholder="Enter text for Video Model 2..."
        )
        output_video_2 = gr.Video()
        generate_button_2 = gr.Button("Generate Video")

if __name__ == "__main__":
    demo.launch()
