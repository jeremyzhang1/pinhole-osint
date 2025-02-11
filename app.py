from typing import List
from pathlib import Path
from functools import partial
import spaces
import gradio as gr
import numpy as np
import torch
from torchvision.datasets.utils import download_and_extract_archive
from PIL import Image
from omegaconf import OmegaConf
from algorithms.dfot import DFoTVideoPose
from algorithms.dfot.history_guidance import HistoryGuidance
from utils.ckpt_utils import download_pretrained
from utils.huggingface_utils import download_from_hf
from datasets.video.utils.io import read_video
from datasets.video import RealEstate10KAdvancedVideoDataset
from export import export_to_video, export_to_gif, export_images_to_gif

DATASET_URL = "https://huggingface.co/kiwhansong/DFoT/resolve/main/datasets/RealEstate10K_Tiny.tar.gz"
DATASET_DIR = Path("data/real-estate-10k-tiny")
LONG_LENGTH = 20  # seconds

if not DATASET_DIR.exists():
    DATASET_DIR.mkdir(parents=True)
    download_and_extract_archive(
        DATASET_URL,
        DATASET_DIR.parent,
        remove_finished=True,
    )


metadata = torch.load(DATASET_DIR / "metadata" / "test.pt", weights_only=False)
video_list = [
    read_video(path).permute(0, 3, 1, 2) / 255.0 for path in metadata["video_paths"]
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

first_frame_list = [
    (video[0] * 255).permute(1, 2, 0).numpy().clip(0, 255).astype("uint8")
    for video in video_list
]
gif_paths = []
for idx, video, path in zip(
    range(len(video_list)), video_list, metadata["video_paths"]
):
    indices = torch.linspace(0, video.size(0) - 1, 16, dtype=torch.long)
    gif_paths.append(export_to_gif(video[indices], fps=8))


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


def prepare_short_gt_video(idx: int):
    video = video_list[idx]
    indices = torch.linspace(0, video.size(0) - 1, 8, dtype=torch.long)
    video = (
        (video[indices].permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8).numpy()
    )
    return [video[i] for i in range(video.shape[0])]


def video_to_gif_and_images(video, indices):
    masked_video = [
        image if i in indices else np.zeros_like(image) for i, image in enumerate(video)
    ]
    return [(export_images_to_gif(masked_video), "GIF")] + [
        (image, f"t={i}" if i in indices else "")
        for i, image in enumerate(masked_video)
    ]


@spaces.GPU(duration=300)
@torch.autocast("cuda")
@torch.no_grad()
def single_image_to_long_video(
    idx: int, guidance_scale: float, fps: int, progress=gr.Progress(track_tqdm=True)
):
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


@torch.autocast("cuda")
@torch.no_grad()
def any_images_to_short_video(
    scene_idx: int,
    image_indices: List[int],
    guidance_scale: float,
    progress=gr.Progress(track_tqdm=True),
):
    video = video_list[scene_idx]
    poses = poses_list[scene_idx]
    indices = torch.linspace(0, video.size(0) - 1, 8, dtype=torch.long)
    xs = video[indices].unsqueeze(0).to("cuda")
    conditions = poses[indices].unsqueeze(0).to("cuda")
    gen_video = dfot._unnormalize_x(
        dfot._sample_sequence(
            batch_size=1,
            context=dfot._normalize_x(xs),
            context_mask=torch.tensor([i in image_indices for i in range(8)])
            .unsqueeze(0)
            .to("cuda"),
            conditions=conditions,
            history_guidance=HistoryGuidance.vanilla(
                guidance_scale=guidance_scale,
                visualize=False,
            ),
        )[0]
    )
    gen_video = (
        (gen_video[0].detach().cpu().permute(0, 2, 3, 1) * 255)
        .clamp(0, 255)
        .to(torch.uint8)
        .numpy()
    )
    return video_to_gif_and_images([image for image in gen_video], list(range(8)))


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
        gr.Button(value="🌐 Website", link="https://boyuan.space/history-guidance")
        gr.Button(value="📄 Paper", link="https://arxiv.org/abs/2502.06764")
        gr.Button(
            value="💻 Code",
            link="https://github.com/kwsong0113/diffusion-forcing-transformer",
        )
        gr.Button(
            value="🤗 Pretrained Models", link="https://huggingface.co/kiwhansong/DFoT"
        )

    with gr.Accordion("Troubleshooting: Not Working or Too Slow?", open=False):
        gr.Markdown(
            """
            - Error or Unexpected Results? _Please try again after refreshing the page._
            - Performance Issues or No GPU Allocation? _Consider running the demo locally by cloning the repository (click the dots in the top-right corner)_.
            """
        )


    with gr.Tab("Any # of Images → Short Video", id="task-1"):
        gr.Markdown(
            """
            ## Demo 1: Any Number of Images → Short 2-second Video
            > #### _Diffusion Forcing Transformer is a flexible model that can generate videos given variable number of context frames._
        """
        )

        demo1_stage = gr.State(value="Scene")
        demo1_selected_scene_index = gr.State(value=None)
        demo1_selected_image_indices = gr.State(value=[])

        @gr.render(
            inputs=[
                demo1_stage,
                demo1_selected_scene_index,
                demo1_selected_image_indices,
            ]
        )
        def render_stage(s, scene_idx, image_indices):
            match s:
                case "Scene":
                    with gr.Group():
                        demo1_scene_gallery = gr.Gallery(
                            height=300,
                            value=gif_paths,
                            label="Select a Scene to Generate Video",
                            columns=[8],
                            selected_index=scene_idx,
                        )

                        @demo1_scene_gallery.select(
                            inputs=None, outputs=demo1_selected_scene_index
                        )
                        def update_selection(selection: gr.SelectData):
                            return selection.index

                        demo1_scene_select_button = gr.Button("Select Scene")

                        @demo1_scene_select_button.click(
                            inputs=demo1_selected_scene_index, outputs=demo1_stage
                        )
                        def move_to_image_selection(scene_idx: int):
                            if scene_idx is None:
                                gr.Warning("Scene not selected!")
                                return "Scene"
                            else:
                                return "Image"

                case "Image":
                    with gr.Group():
                        demo1_image_gallery = gr.Gallery(
                            height=150,
                            value=[
                                (image, f"t={i}")
                                for i, image in enumerate(
                                    prepare_short_gt_video(scene_idx)
                                )
                            ],
                            label="Select Images to Animate",
                            columns=[8],
                        )

                        demo1_selector = gr.CheckboxGroup(
                            label="Select Any Number of Input Images",
                            info="Image-to-Video: Select t=0; Interpolation: Select t=0 and t=7",
                            choices=[(f"t={i}", i) for i in range(8)],
                            value=[],
                        )
                        demo1_image_select_button = gr.Button("Select Input Images")

                        @demo1_image_select_button.click(
                            inputs=[demo1_selector],
                            outputs=[demo1_stage, demo1_selected_image_indices],
                        )
                        def generate_video(selected_indices):
                            if len(selected_indices) == 0:
                                gr.Warning("Select at least one image!")
                                return "Image", []
                            else:
                                return "Generation", selected_indices

                case "Generation":
                    with gr.Group():
                        gt_video = prepare_short_gt_video(scene_idx)

                        demo1_input_image_gallery = gr.Gallery(
                            height=150,
                            value=video_to_gif_and_images(gt_video, image_indices),
                            label="Input Images",
                            columns=[9],
                        )
                        demo1_generated_gallery = gr.Gallery(
                            height=150,
                            value=[],
                            label="Generated Video",
                            columns=[9],
                        )

                        demo1_ground_truth_gallery = gr.Gallery(
                            height=150,
                            value=video_to_gif_and_images(gt_video, list(range(8))),
                            label="Ground Truth Video",
                            columns=[9],
                        )
                    with gr.Sidebar():
                        gr.Markdown("### Sampling Parameters")
                        demo1_guidance_scale = gr.Slider(
                            minimum=1,
                            maximum=6,
                            value=4,
                            step=0.5,
                            label="History Guidance Scale",
                            info="Without history guidance: 1.0; Recommended: 4.0",
                            interactive=True,
                        )
                        gr.Button("Generate Video").click(
                            fn=any_images_to_short_video,
                            inputs=[
                                demo1_selected_scene_index,
                                demo1_selected_image_indices,
                                demo1_guidance_scale,
                            ],
                            outputs=demo1_generated_gallery,
                        )

    with gr.Tab("Single Image → Long Video", id="task-2"):
        gr.Markdown(
            """
            ## Demo 2: Single Image → Long 20-second Video
            > #### _Diffusion Forcing Transformer, with History Guidance, can generate long videos via sliding window rollouts and temporal super-resolution._
        """
        )

        demo2_stage = gr.State(value="Selection")
        demo2_selected_index = gr.State(value=None)

        @gr.render(inputs=[demo2_stage, demo2_selected_index])
        def render_stage(s, idx):
            match s:
                case "Selection":
                    with gr.Group():
                        demo2_image_gallery = gr.Gallery(
                            height=300,
                            value=first_frame_list,
                            label="Select an Image to Animate",
                            columns=[8],
                            selected_index=idx,
                        )

                        @demo2_image_gallery.select(
                            inputs=None, outputs=demo2_selected_index
                        )
                        def update_selection(selection: gr.SelectData):
                            return selection.index

                        demo2_select_button = gr.Button("Select Input Image")

                        @demo2_select_button.click(
                            inputs=demo2_selected_index, outputs=demo2_stage
                        )
                        def move_to_generation(idx: int):
                            if idx is None:
                                gr.Warning("Image not selected!")
                                return "Selection"
                            else:
                                return "Generation"

                case "Generation":
                    with gr.Row():
                        gr.Image(
                            value=first_frame_list[idx],
                            label="Input Image",
                            width=256,
                            height=256,
                        )
                        gr.Video(
                            value=prepare_long_gt_video(idx),
                            label="Ground Truth Video",
                            width=256,
                            height=256,
                        )
                        demo2_video = gr.Video(
                            label="Generated Video", width=256, height=256
                        )

                        with gr.Sidebar():
                            gr.Markdown("### Sampling Parameters")

                            demo2_guidance_scale = gr.Slider(
                                minimum=1,
                                maximum=6,
                                value=4,
                                step=0.5,
                                label="History Guidance Scale",
                                info="Without history guidance: 1.0; Recommended: 4.0",
                                interactive=True,
                            )
                            demo2_fps = gr.Slider(
                                minimum=2,
                                maximum=10,
                                value=4,
                                step=1,
                                label="FPS",
                                info=f"A {LONG_LENGTH}-second video will be generated at this FPS; Decrease for faster generation; Increase for a smoother video",
                                interactive=True,
                            )
                            gr.Button("Generate Video").click(
                                fn=single_image_to_long_video,
                                inputs=[
                                    demo2_selected_index,
                                    demo2_guidance_scale,
                                    demo2_fps,
                                ],
                                outputs=demo2_video,
                            )

    with gr.Tab("Single Image → Extremely Long Video", id="task-3"):
        gr.Markdown(
            """
            ## Demo 3: Single Image → Extremely Long Video
            > #### _TODO._
        """
        )

if __name__ == "__main__":
    demo.launch()
