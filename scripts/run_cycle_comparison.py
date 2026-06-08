from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import nbformat
import numpy as np
from nbclient import NotebookClient
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = PROJECT_ROOT / "notebooks"
OUTPUT_ROOT = PROJECT_ROOT / "data" / "outputs" / "comparison_eval"
RUN_INPUT_ROOT = OUTPUT_ROOT / "inputs"
EXECUTED_NOTEBOOK_ROOT = OUTPUT_ROOT / "executed_notebooks"


MAIN_NOTEBOOKS = [
    NOTEBOOK_ROOT / "01_enhanced_dog_replacement.ipynb",
    NOTEBOOK_ROOT / "02_pose_guided_warp_and_paste.ipynb",
    NOTEBOOK_ROOT / "03_diffusion_refinement.ipynb",
    NOTEBOOK_ROOT / "04_post_generation_harmonization.ipynb",
]

FACE_DRAGGAN_NOTEBOOKS = [
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "01_cutout.ipynb",
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "02_true_draggan_face_inversion.ipynb",
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "03_true_draggan_face_edit.ipynb",
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "04_reinsert_face_then_body_warp.ipynb",
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "05_diffusion_refinement.ipynb",
    NOTEBOOK_ROOT / "face_draggan_pipeline" / "06_post_generation_harmonization.ipynb",
]


@dataclass(frozen=True)
class ExamplePair:
    pair_id: str
    org_path: Path
    rep_path: Path


@dataclass(frozen=True)
class Variant:
    name: str
    group: str
    pipeline: str = "main"
    segmentation_strategy: str = "current_conservative_refined"
    use_matting_anything: bool = True
    pose_alignment_mode: str = "micro_piecewise"
    inpaint_model_preset: str = "realisticvision15_inpaint"
    run_harmonization: bool = True
    generation_include_dog_edge: bool = True
    micro_piecewise: bool = True
    destination_scale: float = 1.03
    background_hole_expansion_px: int = 8
    face_inversion_steps: int | None = None
    face_pti_steps: int | None = None
    face_crop_scale: float | None = None
    face_target_mode: str | None = None
    face_max_support_displacement_px: float | None = None
    face_body_alignment_mode: str = "body_warp"


VARIANTS: dict[str, Variant] = {
    "seg_yolo_only": Variant(
        name="seg_yolo_only",
        group="A",
        segmentation_strategy="yolo_seg_only",
        use_matting_anything=False,
    ),
    "seg_sam_bbox_prior": Variant(
        name="seg_sam_bbox_prior",
        group="A",
        segmentation_strategy="sam_bbox_prior",
        use_matting_anything=False,
    ),
    "seg_yolo_sam_consensus": Variant(
        name="seg_yolo_sam_consensus",
        group="A",
        segmentation_strategy="yolo_sam_consensus",
        use_matting_anything=False,
    ),
    "seg_current_conservative": Variant(
        name="seg_current_conservative",
        group="A",
        segmentation_strategy="current_conservative_refined",
        use_matting_anything=True,
    ),
    "pose_no_adjustment": Variant(
        name="pose_no_adjustment",
        group="B",
        pose_alignment_mode="naive_box",
        micro_piecewise=False,
    ),
    "pose_micro_piecewise": Variant(
        name="pose_micro_piecewise",
        group="B",
        pose_alignment_mode="micro_piecewise",
        micro_piecewise=True,
    ),
    "pose_face_only": Variant(
        name="pose_face_only",
        group="B",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
        face_body_alignment_mode="face_only_naive_box",
    ),
    "pose_face_draggan_current": Variant(
        name="pose_face_draggan_current",
        group="B",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
        face_body_alignment_mode="body_warp",
    ),
    "diff_realisticvision15": Variant(
        name="diff_realisticvision15",
        group="C",
        inpaint_model_preset="realisticvision15_inpaint",
        run_harmonization=True,
        generation_include_dog_edge=True,
    ),
    "diff_realvisxl": Variant(
        name="diff_realvisxl",
        group="C",
        inpaint_model_preset="realvisxl_inpaint",
        run_harmonization=True,
        generation_include_dog_edge=True,
    ),
    "diff_sdxl": Variant(
        name="diff_sdxl",
        group="C",
        inpaint_model_preset="sdxl_inpaint",
        run_harmonization=True,
        generation_include_dog_edge=True,
    ),
    "harmonization_enabled": Variant(
        name="harmonization_enabled",
        group="D",
        inpaint_model_preset="realisticvision15_inpaint",
        run_harmonization=True,
        generation_include_dog_edge=True,
    ),
    "harmonization_disabled": Variant(
        name="harmonization_disabled",
        group="D",
        inpaint_model_preset="realisticvision15_inpaint",
        run_harmonization=False,
        generation_include_dog_edge=True,
    ),
    "face_draggan_current": Variant(
        name="face_draggan_current",
        group="E",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
    ),
    "face_draggan_fast_pti": Variant(
        name="face_draggan_fast_pti",
        group="E",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
        face_inversion_steps=450,
        face_pti_steps=250,
    ),
    "face_draggan_high_inversion": Variant(
        name="face_draggan_high_inversion",
        group="E",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
        face_inversion_steps=1600,
        face_pti_steps=600,
        face_crop_scale=1.45,
    ),
    "face_draggan_lower_support": Variant(
        name="face_draggan_lower_support",
        group="E",
        pipeline="face_draggan",
        face_target_mode="nose_drag_with_structure",
        face_max_support_displacement_px=45.0,
    ),
}

# Backward-compatible aliases from earlier pilot runs.
VARIANTS["current_main"] = VARIANTS["diff_realisticvision15"]
VARIANTS["no_harmonization"] = VARIANTS["harmonization_disabled"]
VARIANTS["no_dog_edge_generation"] = Variant(
    name="no_dog_edge_generation",
    group="C",
    inpaint_model_preset="realisticvision15_inpaint",
    run_harmonization=True,
    generation_include_dog_edge=False,
)
VARIANTS["realvisxl"] = VARIANTS["diff_realvisxl"]
VARIANTS["sdxl"] = VARIANTS["diff_sdxl"]

REPORT_GROUPS = [
    {
        "group": "A. Segmentation Strategy Ablation",
        "description": "Compares the four segmentation strategies from comparison.md while keeping geometry, diffusion, and harmonization fixed.",
        "variants": ["seg_yolo_only", "seg_sam_bbox_prior", "seg_yolo_sam_consensus", "seg_current_conservative"],
    },
    {
        "group": "B. Pose / Geometry Adjustment Ablation",
        "description": "Compares active geometry strategies only: face adjustment only, body pose adjustment only, and face + body pose adjustment.",
        "variants": ["pose_face_only", "pose_micro_piecewise", "pose_face_draggan_current"],
    },
    {
        "group": "C. Diffusion Model Ablation",
        "description": "Compares the three inpainting model presets while keeping segmentation, geometry, and harmonization fixed.",
        "variants": ["diff_realisticvision15", "current_main", "diff_realvisxl", "realvisxl", "diff_sdxl", "sdxl"],
    },
    {
        "group": "D. Post-Generation Harmonization Ablation",
        "description": "Tests whether notebook 04 LAB harmonization/light sharpening improves cycle consistency.",
        "variants": ["harmonization_enabled", "current_main", "harmonization_disabled", "no_harmonization"],
    },
    {
        "group": "E. Face-DragGAN Parameter Ablation",
        "description": "Focused non-default Face-DragGAN tuning variants from comparison.md. These are expensive and can be run separately with --groups E.",
        "variants": ["face_draggan_fast_pti", "face_draggan_high_inversion", "face_draggan_lower_support"],
    },
]

PRIMARY_SCORE_METRICS = [
    ("edited_region_lpips", "lower"),
    ("boundary_ring_mae", "lower"),
    ("background_mae", "lower"),
]

LPIPS_MODEL = None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def read_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_mask(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L")) > 127
    if shape_hw is not None and mask.shape[:2] != shape_hw:
        mask = cv2.resize(mask.astype(np.uint8), (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def get_lpips_model():
    global LPIPS_MODEL
    if LPIPS_MODEL is not None:
        return LPIPS_MODEL
    try:
        import torch
        import lpips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
        return LPIPS_MODEL
    except Exception as exc:
        print(f"[warn] LPIPS unavailable, edited_region_lpips will be NaN: {exc!r}")
        LPIPS_MODEL = False
        return None


def masked_lpips(a: np.ndarray, b: np.ndarray, mask: np.ndarray, pad: int = 24) -> float:
    model = get_lpips_model()
    if model is None or model is False or not mask.any():
        return float("nan")

    import torch

    ys, xs = np.where(mask)
    y1 = max(int(ys.min()) - pad, 0)
    y2 = min(int(ys.max()) + pad + 1, a.shape[0])
    x1 = max(int(xs.min()) - pad, 0)
    x2 = min(int(xs.max()) + pad + 1, a.shape[1])
    a_crop = a[y1:y2, x1:x2]
    b_crop = b[y1:y2, x1:x2]
    if min(a_crop.shape[:2]) < 8:
        return float("nan")

    target_size = 256
    a_crop = cv2.resize(a_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    b_crop = cv2.resize(b_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    device = next(model.parameters()).device
    a_tensor = torch.from_numpy(a_crop).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
    b_tensor = torch.from_numpy(b_crop).float().permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0
    with torch.no_grad():
        return float(model(a_tensor, b_tensor).item())


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(path)


def copy_as_png(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image = Image.open(src).convert("RGB")
    image.save(dst)
    return dst


def clear_active_input_state(cutout_root: Path) -> None:
    """Notebook 01 must use the scripted env inputs, not a stale manual upload."""
    for filename in ["_active_session.json", "_active_batch_manifest.json"]:
        path = cutout_root / filename
        if path.exists():
            path.unlink()


def discover_pairs() -> list[ExamplePair]:
    candidates = [
        ("pair_1", PROJECT_ROOT / "data" / "example" / "pair 1" / "org1.png", PROJECT_ROOT / "data" / "example" / "pair 1" / "rep1.jpg"),
        ("pair_2", PROJECT_ROOT / "data" / "example" / "pari 2" / "org.jpg", PROJECT_ROOT / "data" / "example" / "pari 2" / "rep.jpg"),
        ("face_pair", PROJECT_ROOT / "data" / "example" / "face_pair" / "org.jpg", PROJECT_ROOT / "data" / "example" / "face_pair" / "rep.png"),
    ]
    pairs = []
    for pair_id, org_path, rep_path in candidates:
        if org_path.exists() and rep_path.exists():
            pairs.append(ExamplePair(pair_id=pair_id, org_path=org_path, rep_path=rep_path))
    if not pairs:
        raise FileNotFoundError("No example pairs were found under data/example.")
    return pairs


def replace_assignment(source: str, name: str, value: Any) -> str:
    if isinstance(value, str):
        rendered = json.dumps(value)
    elif isinstance(value, bool):
        rendered = "True" if value else "False"
    else:
        rendered = repr(value)
    pattern = re.compile(rf"^({re.escape(name)}\s*=\s*).*$", flags=re.MULTILINE)
    if not pattern.search(source):
        return source
    return pattern.sub(rf"\g<1>{rendered}", source, count=1)


def patched_notebook(path: Path, replacements: dict[str, Any]) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        for name, value in replacements.items():
            source = replace_assignment(source, name, value)
        cell.source = source
    return notebook


def execute_notebook(
    notebook_path: Path,
    executed_path: Path,
    env: dict[str, str],
    replacements: dict[str, Any] | None = None,
    timeout: int = 7200,
) -> None:
    old_env = os.environ.copy()
    os.environ.update(env)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        notebook = patched_notebook(notebook_path, replacements or {})
        client = NotebookClient(
            notebook,
            timeout=timeout,
            startup_timeout=max(300, min(timeout, 1800)),
            kernel_name="python3",
            resources={"metadata": {"path": str(PROJECT_ROOT)}},
            allow_errors=False,
        )
        client.execute()
        executed_path.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(notebook, executed_path)
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def pair_name_from_paths(org_path: Path, rep_path: Path) -> str:
    return f"{org_path.stem}__{rep_path.stem}"


def run_pipeline_once(
    *,
    org_path: Path,
    rep_path: Path,
    pair_slug: str,
    direction: str,
    variant: Variant,
    timeout: int,
) -> dict[str, Any]:
    route = f"eval_{variant.name}_{pair_slug}_{direction}"
    env = {
        "DOG_REPLACEMENT_ORG_PATH": str(org_path.resolve()),
        "DOG_REPLACEMENT_REP_PATH": str(rep_path.resolve()),
        "DOG_REFINER_SEED": "7",
        "DOG_SEGMENTATION_STRATEGY": variant.segmentation_strategy,
        "DOG_USE_MATTING_ANYTHING": "1" if variant.use_matting_anything else "0",
    }
    log_dir = EXECUTED_NOTEBOOK_ROOT / variant.name / pair_slug / direction
    print(f"\n[run] {variant.name} / {pair_slug} / {direction}")
    print(f"      org={rel(org_path)}")
    print(f"      rep={rel(rep_path)}")

    clear_active_input_state(PROJECT_ROOT / "data" / "outputs" / "01_cutout")
    execute_notebook(
        MAIN_NOTEBOOKS[0],
        log_dir / "01_executed.ipynb",
        env=env,
        timeout=timeout,
    )
    cutout_manifest_path = PROJECT_ROOT / "data" / "outputs" / "01_cutout" / "_active_batch_manifest.json"
    cutout_manifest = read_json(cutout_manifest_path)
    if not cutout_manifest:
        raise RuntimeError(f"Notebook 01 did not write an active manifest: {cutout_manifest_path}")
    pair_name = cutout_manifest[0]["pair_name"]
    cutout_meta_path = PROJECT_ROOT / cutout_manifest[0]["metadata_path"]

    execute_notebook(
        MAIN_NOTEBOOKS[1],
        log_dir / "02_executed.ipynb",
        env={"DOG_POSE_ALIGNMENT_MODE": variant.pose_alignment_mode},
        replacements={
            "POSE_ALIGNMENT_MODE": variant.pose_alignment_mode,
            "ENABLE_MICRO_PIECEWISE_WARP": variant.micro_piecewise,
            "ENABLE_FULL_PIECEWISE_WARP": False,
            "REPLACEMENT_DESTINATION_SCALE": variant.destination_scale,
            "BACKGROUND_HOLE_EXPANSION_PX": variant.background_hole_expansion_px,
        },
        timeout=timeout,
    )
    execute_notebook(
        MAIN_NOTEBOOKS[2],
        log_dir / "03_executed.ipynb",
        env={},
        replacements={
            "ROUTE_MODE": route,
            "INPAINT_MODEL_PRESET": variant.inpaint_model_preset,
            "GENERATION_MASK_INCLUDE_DOG_EDGE": variant.generation_include_dog_edge,
        },
        timeout=timeout,
    )

    refine_manifest_path = PROJECT_ROOT / "data" / "outputs" / "03_diffusion_refine" / route / "batch_manifest.json"
    refine_manifest = read_json(refine_manifest_path)
    refine_meta_path = Path(refine_manifest[0]["run_config_path"])
    refine_meta = read_json(refine_meta_path)
    final_path = Path(refine_meta["outputs"]["refined_final"])
    final_stage = "03_diffusion"

    if variant.run_harmonization:
        execute_notebook(
            MAIN_NOTEBOOKS[3],
            log_dir / "04_executed.ipynb",
            env={},
            replacements={"ROUTE_MODE": route},
            timeout=timeout,
        )
        post_manifest_path = PROJECT_ROOT / "data" / "outputs" / "04_postprocess" / route / "batch_manifest.json"
        post_manifest = read_json(post_manifest_path)
        post_meta_path = Path(post_manifest[0].get("metadata_path") or post_manifest[0]["run_summary_path"])
        post_meta = read_json(post_meta_path)
        final_path = Path(post_meta["outputs"]["postprocessed_final"])
        final_stage = "04_postprocess"
    else:
        post_manifest_path = None
        post_meta_path = None

    warp_meta_path = PROJECT_ROOT / "data" / "outputs" / "02_pose_warp" / pair_name / "metadata.json"
    return {
        "pair_name": pair_name,
        "route": route,
        "final_path": final_path,
        "final_stage": final_stage,
        "cutout_meta_path": cutout_meta_path,
        "warp_meta_path": warp_meta_path,
        "refine_manifest_path": refine_manifest_path,
        "post_manifest_path": post_manifest_path,
        "post_meta_path": post_meta_path,
    }


def run_face_draggan_pipeline_once(
    *,
    org_path: Path,
    rep_path: Path,
    pair_slug: str,
    direction: str,
    variant: Variant,
    timeout: int,
) -> dict[str, Any]:
    route = f"eval_{variant.name}_{pair_slug}_{direction}"
    env = {
        "DOG_REPLACEMENT_ORG_PATH": str(org_path.resolve()),
        "DOG_REPLACEMENT_REP_PATH": str(rep_path.resolve()),
        "DOG_REFINER_SEED": "7",
        "DOG_TRUE_DRAGGAN_MAX_PAIRS": "1",
        "DOG_TRUE_DRAGGAN_FACE_TARGET_MODE": variant.face_target_mode or "nose_drag_with_structure",
        "DOG_TRUE_DRAGGAN_BODY_ALIGNMENT_MODE": variant.face_body_alignment_mode,
    }
    if variant.face_inversion_steps is not None:
        env["DOG_TRUE_DRAGGAN_INVERSION_STEPS"] = str(int(variant.face_inversion_steps))
    if variant.face_pti_steps is not None:
        env["DOG_TRUE_DRAGGAN_PTI_STEPS"] = str(int(variant.face_pti_steps))
    if variant.face_crop_scale is not None:
        env["DOG_TRUE_DRAGGAN_FACE_CROP_SCALE"] = str(float(variant.face_crop_scale))
    if variant.face_max_support_displacement_px is not None:
        env["DOG_TRUE_DRAGGAN_NOSE_DRAG_MAX_SUPPORT_DISPLACEMENT_PX"] = str(float(variant.face_max_support_displacement_px))

    log_dir = EXECUTED_NOTEBOOK_ROOT / variant.name / pair_slug / direction
    print(f"\n[run-face] {variant.name} / {pair_slug} / {direction}")
    print(f"           org={rel(org_path)}")
    print(f"           rep={rel(rep_path)}")

    clear_active_input_state(PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "01_cutout")
    execute_notebook(FACE_DRAGGAN_NOTEBOOKS[0], log_dir / "01_face_executed.ipynb", env=env, timeout=timeout)
    cutout_manifest_path = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "01_cutout" / "_active_batch_manifest.json"
    cutout_manifest = read_json(cutout_manifest_path)
    if not cutout_manifest:
        raise RuntimeError(f"Face notebook 01 did not write an active manifest: {cutout_manifest_path}")
    pair_name = cutout_manifest[0]["pair_name"]
    cutout_meta_path = PROJECT_ROOT / cutout_manifest[0]["metadata_path"]

    execute_notebook(FACE_DRAGGAN_NOTEBOOKS[1], log_dir / "02_face_inversion_executed.ipynb", env=env, timeout=timeout)
    execute_notebook(FACE_DRAGGAN_NOTEBOOKS[2], log_dir / "03_face_edit_executed.ipynb", env=env, timeout=timeout)
    execute_notebook(FACE_DRAGGAN_NOTEBOOKS[3], log_dir / "04_face_body_warp_executed.ipynb", env=env, timeout=timeout)
    execute_notebook(
        FACE_DRAGGAN_NOTEBOOKS[4],
        log_dir / "05_face_diffusion_executed.ipynb",
        env=env,
        replacements={
            "ROUTE_MODE": route,
            "INPAINT_MODEL_PRESET": variant.inpaint_model_preset,
            "GENERATION_MASK_INCLUDE_DOG_EDGE": variant.generation_include_dog_edge,
        },
        timeout=timeout,
    )

    refine_manifest_path = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "05_diffusion_refine" / route / "batch_manifest.json"
    refine_manifest = read_json(refine_manifest_path)
    refine_meta_path = Path(refine_manifest[0]["run_config_path"])
    refine_meta = read_json(refine_meta_path)
    final_path = Path(refine_meta["outputs"]["refined_final"])
    final_stage = "face_05_diffusion"
    post_manifest_path = None
    post_meta_path = None

    if variant.run_harmonization:
        execute_notebook(
            FACE_DRAGGAN_NOTEBOOKS[5],
            log_dir / "06_face_postprocess_executed.ipynb",
            env=env,
            replacements={"ROUTE_MODE": route},
            timeout=timeout,
        )
        post_manifest_path = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "06_postprocess" / route / "batch_manifest.json"
        post_manifest = read_json(post_manifest_path)
        post_meta_path = Path(post_manifest[0].get("metadata_path") or post_manifest[0]["run_summary_path"])
        post_meta = read_json(post_meta_path)
        final_path = Path(post_meta["outputs"]["postprocessed_final"])
        final_stage = "face_06_postprocess"

    warp_meta_path = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "04_body_warp_after_true_draggan" / pair_name / "metadata.json"
    return {
        "pair_name": pair_name,
        "route": route,
        "final_path": final_path,
        "final_stage": final_stage,
        "cutout_meta_path": cutout_meta_path,
        "warp_meta_path": warp_meta_path,
        "refine_manifest_path": refine_manifest_path,
        "post_manifest_path": post_manifest_path,
        "post_meta_path": post_meta_path,
    }


def gaussian_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    a_f = a.astype(np.float32)
    b_f = b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    scores = []
    for ch in range(3):
        x = a_f[..., ch]
        y = b_f[..., ch]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        if mask is not None and mask.any():
            scores.append(float(np.mean(ssim_map[mask])))
        else:
            scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def psnr(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (a.astype(np.float32) - b.astype(np.float32)) ** 2
    if mask is not None and mask.any():
        mse = float(np.mean(diff[mask]))
    else:
        mse = float(np.mean(diff))
    if mse <= 1e-10:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    if mask is not None and mask.any():
        return float(np.mean(diff[mask]))
    return float(np.mean(diff))


def build_regions(mask: np.ndarray) -> dict[str, np.ndarray]:
    mask_u8 = mask.astype(np.uint8)
    dilate_31 = cv2.dilate(mask_u8, np.ones((31, 31), np.uint8), iterations=1) > 0
    dilate_65 = cv2.dilate(mask_u8, np.ones((65, 65), np.uint8), iterations=1) > 0
    erode_9 = cv2.erode(mask_u8, np.ones((9, 9), np.uint8), iterations=1) > 0
    return {
        "dog_region": mask,
        "boundary_ring": np.logical_and(dilate_31, ~erode_9),
        "edited_region": dilate_65,
        "background": ~dilate_65,
    }


def compute_metrics(original_path: Path, reconstructed_path: Path, mask_path: Path) -> dict[str, float]:
    original = read_rgb(original_path)
    reconstructed = read_rgb(reconstructed_path)
    if reconstructed.shape[:2] != original.shape[:2]:
        reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC)
    mask = read_mask(mask_path, original.shape[:2])
    regions = build_regions(mask)
    metrics: dict[str, float] = {
        "full_image_mae": mae(original, reconstructed),
        "full_image_psnr": psnr(original, reconstructed),
        "full_image_ssim": gaussian_ssim(original, reconstructed),
    }
    for region_name, region_mask in regions.items():
        if not region_mask.any():
            metrics[f"{region_name}_mae"] = float("nan")
            metrics[f"{region_name}_psnr"] = float("nan")
            metrics[f"{region_name}_ssim"] = float("nan")
            continue
        metrics[f"{region_name}_mae"] = mae(original, reconstructed, region_mask)
        metrics[f"{region_name}_psnr"] = psnr(original, reconstructed, region_mask)
        metrics[f"{region_name}_ssim"] = gaussian_ssim(original, reconstructed, region_mask)
    metrics["edited_region_lpips"] = masked_lpips(original, reconstructed, regions["edited_region"])
    metrics["active_region_mae"] = metrics["edited_region_mae"]
    metrics["active_region_psnr"] = metrics["edited_region_psnr"]
    metrics["active_region_ssim"] = metrics["edited_region_ssim"]
    metrics["global_mae"] = metrics["active_region_mae"]
    metrics["global_psnr"] = metrics["active_region_psnr"]
    metrics["global_ssim"] = metrics["active_region_ssim"]
    return metrics


def run_cycle(pair: ExamplePair, variant: Variant, timeout: int, skip_existing: bool) -> dict[str, Any]:
    pair_slug = pair.pair_id
    run_dir = OUTPUT_ROOT / "runs" / variant.name / pair_slug
    summary_path = run_dir / "cycle_summary.json"
    if skip_existing and summary_path.exists():
        summary = read_json(summary_path)
        artifacts = summary.get("artifacts", {})
        if artifacts:
            summary["metrics"] = compute_metrics(
                Path(artifacts["reference_original"]),
                Path(artifacts["reverse_final"]),
                Path(artifacts["original_mask"]),
            )
            write_json(summary_path, summary)
        return summary

    input_dir = RUN_INPUT_ROOT / variant.name / pair_slug
    forward_org = copy_as_png(pair.org_path, input_dir / f"{pair_slug}_forward_org.png")
    forward_rep = copy_as_png(pair.rep_path, input_dir / f"{pair_slug}_forward_rep.png")
    original_copy = copy_as_png(pair.org_path, input_dir / f"{pair_slug}_reference_original.png")

    started = time.time()
    runner = run_face_draggan_pipeline_once if variant.pipeline == "face_draggan" else run_pipeline_once

    forward = runner(
        org_path=forward_org,
        rep_path=forward_rep,
        pair_slug=pair_slug,
        direction="forward",
        variant=variant,
        timeout=timeout,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    forward_cutout_meta = read_json(forward["cutout_meta_path"])
    original_mask_path = PROJECT_ROOT / forward_cutout_meta["original"]["mask_path"]
    preserved_original_mask_path = run_dir / "forward_original_mask.png"
    shutil.copy2(original_mask_path, preserved_original_mask_path)
    forward_final_snapshot = copy_as_png(forward["final_path"], run_dir / "forward_final.png")

    forward_final_for_reverse = copy_as_png(
        forward_final_snapshot,
        input_dir / f"{pair_slug}_reverse_org_forward_result.png",
    )
    reverse_rep = copy_as_png(pair.org_path, input_dir / f"{pair_slug}_reverse_rep_original_dog.png")
    reverse = runner(
        org_path=forward_final_for_reverse,
        rep_path=reverse_rep,
        pair_slug=pair_slug,
        direction="reverse",
        variant=variant,
        timeout=timeout,
    )
    elapsed = time.time() - started
    reverse_final_snapshot = copy_as_png(reverse["final_path"], run_dir / "reverse_final.png")

    metrics = compute_metrics(original_copy, reverse_final_snapshot, preserved_original_mask_path)
    cycle_sanity = {
        "reverse_vs_forward_mae": mae(read_rgb(reverse_final_snapshot), read_rgb(forward_final_snapshot)),
        "reverse_vs_reference_mae": mae(read_rgb(reverse_final_snapshot), read_rgb(original_copy)),
    }

    artifacts = {
        "forward_final": str(forward_final_snapshot.resolve()),
        "reverse_final": str(reverse_final_snapshot.resolve()),
        "forward_final_source": str(Path(forward["final_path"]).resolve()),
        "reverse_final_source": str(Path(reverse["final_path"]).resolve()),
        "reference_original": str(original_copy.resolve()),
        "original_mask": str(preserved_original_mask_path.resolve()),
    }
    summary = {
        "pair_id": pair.pair_id,
        "variant": variant.name,
        "elapsed_seconds": elapsed,
        "forward": {k: str(v) if isinstance(v, Path) else v for k, v in forward.items()},
        "reverse": {k: str(v) if isinstance(v, Path) else v for k, v in reverse.items()},
        "artifacts": artifacts,
        "cycle_sanity": cycle_sanity,
        "metrics": metrics,
    }
    write_json(summary_path, summary)
    return summary


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "variant": summary["variant"],
        "pair_id": summary["pair_id"],
        "elapsed_min": round(float(summary["elapsed_seconds"]) / 60.0, 3),
        "forward_stage": summary["forward"]["final_stage"],
        "reverse_stage": summary["reverse"]["final_stage"],
        "forward_final": rel(Path(summary["artifacts"]["forward_final"])),
        "reverse_final": rel(Path(summary["artifacts"]["reverse_final"])),
    }
    for key, value in summary["metrics"].items():
        row[key] = round(float(value), 6) if math.isfinite(float(value)) else value
    return row


def value_to_float(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def format_metric(value: Any, digits: int = 3) -> str:
    value_f = value_to_float(value)
    if value_f is None:
        return "N/A"
    return f"{value_f:.{digits}f}"


def score_rows_for_pairs(rows: list[dict[str, Any]], pair_ids: list[str]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], list[float]] = {}
    for pair_id in pair_ids:
        pair_rows = [row for row in rows if row.get("pair_id") == pair_id]
        for metric, direction in PRIMARY_SCORE_METRICS:
            values = [(row, value_to_float(row.get(metric))) for row in pair_rows]
            values = [(row, value) for row, value in values if value is not None]
            if not values:
                continue
            metric_values = [value for _, value in values]
            min_v = min(metric_values)
            max_v = max(metric_values)
            for row, value in values:
                if abs(max_v - min_v) < 1e-12:
                    score = 1.0
                elif direction == "lower":
                    score = (max_v - value) / (max_v - min_v)
                else:
                    score = (value - min_v) / (max_v - min_v)
                scores.setdefault((str(row["variant"]), pair_id), []).append(float(score))

    return {
        key: float(np.mean(metric_scores)) if metric_scores else float("nan")
        for key, metric_scores in scores.items()
    }


def rank_rows(rows: list[dict[str, Any]], pair_ids: list[str]) -> list[dict[str, Any]]:
    pair_scores = score_rows_for_pairs(rows, pair_ids)
    ranked = []
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if str(row["variant"]) == variant]
        variant_pair_scores = [pair_scores.get((variant, pair_id), float("nan")) for pair_id in pair_ids]
        finite_scores = [score for score in variant_pair_scores if math.isfinite(score)]
        mean_score = float(np.mean(finite_scores)) if finite_scores else float("nan")
        ranked.append(
            {
                "variant": variant,
                "rows": variant_rows,
                "pair_scores": dict(zip(pair_ids, variant_pair_scores)),
                "mean_score": mean_score,
                "avg_boundary_ring_mae": mean_finite([r.get("boundary_ring_mae") for r in variant_rows]),
                "avg_edited_region_lpips": mean_finite([r.get("edited_region_lpips") for r in variant_rows]),
                "avg_background_mae": mean_finite([r.get("background_mae") for r in variant_rows]),
                "avg_elapsed_min": mean_finite([r.get("elapsed_min") for r in variant_rows]),
            }
        )
    ranked.sort(key=lambda row: (-row["mean_score"] if math.isfinite(row["mean_score"]) else float("inf"), row["variant"]))
    return ranked


def write_ranked_group_table(f, title: str, description: str, rows: list[dict[str, Any]], pair_ids: list[str]) -> None:
    f.write(f"## {title}\n\n")
    f.write(description.strip() + "\n\n")
    if not rows:
        f.write("No executed variants are available for this group yet.\n\n")
        return

    ranked = rank_rows(rows, pair_ids)
    columns = [
        "rank",
        "variant",
        *[f"{pair_id} score" for pair_id in pair_ids],
        "mean score",
        "edited LPIPS lower",
        "boundary MAE lower",
        "background MAE lower",
        "runtime min",
    ]
    f.write("| " + " | ".join(columns) + " |\n")
    f.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
    for idx, item in enumerate(ranked, start=1):
        values = [
            str(idx),
            item["variant"],
            *[format_metric(item["pair_scores"].get(pair_id), 3) for pair_id in pair_ids],
            format_metric(item["mean_score"], 3),
            format_metric(item["avg_edited_region_lpips"], 4),
            format_metric(item["avg_boundary_ring_mae"], 3),
            format_metric(item["avg_background_mae"], 3),
            format_metric(item["avg_elapsed_min"], 3),
        ]
        f.write("| " + " | ".join(values) + " |\n")
    f.write("\n")


def write_tables(rows: list[dict[str, Any]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "cycle_metrics.csv"
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUTPUT_ROOT / "cycle_metrics.md"
    metric_cols = [
        "variant",
        "pair_id",
        "elapsed_min",
        "edited_region_lpips",
        "boundary_ring_mae",
        "background_mae",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Cycle Consistency Evaluation\n\n")
        f.write(
            "Primary ranking uses three cycle-replacement metrics only: edited-region LPIPS, boundary-ring MAE, "
            "and background MAE. Full-image and SSIM/PSNR metrics are retained in `cycle_metrics.csv` only as secondary diagnostics.\n\n"
        )
        f.write("Pair-score is a normalized composite score within each table: higher is better. It combines edited-region LPIPS, boundary MAE, and background MAE.\n\n")

        preferred_pair_order = ["pair_1", "pair_2", "face_pair"]
        discovered_pair_ids = {str(row["pair_id"]) for row in rows}
        pair_ids = [pair_id for pair_id in preferred_pair_order if pair_id in discovered_pair_ids]
        pair_ids.extend(sorted(discovered_pair_ids - set(pair_ids)))
        for group in REPORT_GROUPS:
            group_rows = [row for row in rows if row["variant"] in group["variants"]]
            write_ranked_group_table(f, group["group"], group["description"], group_rows, pair_ids)

        f.write("## Overall Ranking\n\n")
        f.write("This table ranks every executed variant by average composite score across all evaluated pairs.\n\n")
        write_ranked_group_table(f, "All Executed Variants", "Average over all available comparison rows.", rows, pair_ids)

        f.write("## Per-Pair Raw Metrics\n\n")
        f.write("These are the raw numbers behind the ranking. Lower MAE/LPIPS is better.\n\n")
        f.write("| " + " | ".join(metric_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(metric_cols)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in metric_cols) + " |\n")
        f.write("\nFull artifact paths and extra metrics are in `cycle_metrics.csv`.\n")

    aggregate_rows = aggregate_by_variant(rows)
    aggregate_csv_path = OUTPUT_ROOT / "cycle_metrics_by_variant.csv"
    aggregate_md_path = OUTPUT_ROOT / "cycle_metrics_by_variant.md"
    aggregate_cols = [
        "variant",
        "n_pairs",
        "avg_edited_region_lpips",
        "avg_boundary_ring_mae",
        "avg_background_mae",
        "avg_elapsed_min",
    ]
    with open(aggregate_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_cols)
        writer.writeheader()
        writer.writerows(aggregate_rows)
    with open(aggregate_md_path, "w", encoding="utf-8") as f:
        f.write("# Cycle Consistency Aggregate Metrics\n\n")
        f.write("Averages across all evaluated example pairs. Lower MAE/LPIPS is better.\n\n")
        f.write("| " + " | ".join(aggregate_cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(aggregate_cols)) + " |\n")
        for row in aggregate_rows:
            f.write("| " + " | ".join(str(row.get(col, "")) for col in aggregate_cols) + " |\n")

    print(f"\nSaved metrics CSV: {csv_path}")
    print(f"Saved metrics Markdown: {md_path}")
    print(f"Saved aggregate CSV: {aggregate_csv_path}")
    print(f"Saved aggregate Markdown: {aggregate_md_path}")


def collect_completed_summaries(recompute_metrics: bool = False) -> list[dict[str, Any]]:
    summaries = []
    for summary_path in sorted((OUTPUT_ROOT / "runs").glob("*/*/cycle_summary.json")):
        summary = read_json(summary_path)
        if summary.get("variant") not in VARIANTS:
            continue
        if recompute_metrics:
            artifacts = summary.get("artifacts", {})
            if artifacts:
                summary["metrics"] = compute_metrics(
                    Path(artifacts["reference_original"]),
                    Path(artifacts["reverse_final"]),
                    Path(artifacts["original_mask"]),
                )
                write_json(summary_path, summary)
        summaries.append(summary)
    return summaries


def mean_finite(values: list[Any]) -> float:
    finite_values = []
    for value in values:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            finite_values.append(value_f)
    if not finite_values:
        return float("nan")
    return float(np.mean(finite_values))


def aggregate_by_variant(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)

    aggregate_rows = []
    for variant, variant_rows in sorted(grouped.items()):
        aggregate_rows.append(
            {
                "variant": variant,
                "n_pairs": len(variant_rows),
                "avg_edited_region_lpips": round(mean_finite([r.get("edited_region_lpips") for r in variant_rows]), 6),
                "avg_boundary_ring_mae": round(mean_finite([r.get("boundary_ring_mae") for r in variant_rows]), 6),
                "avg_background_mae": round(mean_finite([r.get("background_mae") for r in variant_rows]), 6),
                "avg_elapsed_min": round(mean_finite([r.get("elapsed_min") for r in variant_rows]), 6),
            }
        )
    return aggregate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run end-to-end dog replacement cycle-consistency evaluation.")
    parser.add_argument(
        "--variants",
        default=None,
        help=f"Comma-separated variants. If omitted, variants are selected from --groups. Available: {', '.join(sorted(VARIANTS))}",
    )
    parser.add_argument(
        "--groups",
        default="A,B,C,D,E",
        help="Comma-separated comparison groups to run when --variants is omitted. Options: A,B,C,D,E.",
    )
    parser.add_argument("--max-pairs", type=int, default=3, help="Maximum number of example pairs to evaluate.")
    parser.add_argument("--pair-ids", default=None, help="Comma-separated pair ids to evaluate, e.g. pair_2,face_pair.")
    parser.add_argument("--timeout", type=int, default=7200, help="Per-notebook timeout in seconds.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse completed cycle summaries if present.")
    parser.add_argument("--list-variants", action="store_true", help="Print available variants and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_variants:
        for name, variant in sorted(VARIANTS.items()):
            print(f"{name}: group={variant.group}, pipeline={variant.pipeline}, model={variant.inpaint_model_preset}")
        return

    selected_variants = []
    if args.variants:
        variant_names = [item.strip() for item in args.variants.split(",") if item.strip()]
    else:
        requested_groups = {item.strip().upper() for item in args.groups.split(",") if item.strip()}
        variant_names = [
            name
            for name, variant in VARIANTS.items()
            if variant.group.upper() in requested_groups and name not in {"current_main", "no_harmonization", "no_dog_edge_generation", "realvisxl", "sdxl"}
        ]

    for name in variant_names:
        if name not in VARIANTS:
            raise KeyError(f"Unknown variant: {name}. Available variants: {sorted(VARIANTS)}")
        selected_variants.append(VARIANTS[name])
    pairs = discover_pairs()
    if args.pair_ids:
        requested_pair_ids = {item.strip() for item in args.pair_ids.split(",") if item.strip()}
        pairs = [pair for pair in pairs if pair.pair_id in requested_pair_ids]
    pairs = pairs[: args.max_pairs]
    if not pairs:
        raise RuntimeError("No pairs selected.")

    print("Selected pairs:")
    for pair in pairs:
        print(f" - {pair.pair_id}: {rel(pair.org_path)} + {rel(pair.rep_path)}")
    print("Selected variants:", ", ".join(v.name for v in selected_variants))

    summaries = []
    failures = []
    for variant in selected_variants:
        for pair in pairs:
            try:
                summaries.append(run_cycle(pair, variant, timeout=args.timeout, skip_existing=args.skip_existing))
            except Exception as exc:
                failures.append({"variant": variant.name, "pair_id": pair.pair_id, "error": repr(exc)})
                write_json(OUTPUT_ROOT / "failures.json", failures)
                print(f"\n[failed] {variant.name} / {pair.pair_id}: {exc!r}")
                raise

    all_summaries = collect_completed_summaries(recompute_metrics=False)
    selected_keys = {(summary["variant"], summary["pair_id"]) for summary in summaries}
    existing_keys = {(summary["variant"], summary["pair_id"]) for summary in all_summaries}
    for summary in summaries:
        if (summary["variant"], summary["pair_id"]) not in existing_keys:
            all_summaries.append(summary)
    rows = [flatten_summary(summary) for summary in all_summaries]
    write_tables(rows)
    print("\nSummary rows:")
    for row in rows:
        print(
            f"{row['variant']} | {row['pair_id']} | boundary MAE={row.get('boundary_ring_mae')} | "
            f"dog SSIM={row.get('dog_region_ssim')} | background MAE={row.get('background_mae')} | "
            f"elapsed={row.get('elapsed_min')} min"
        )


if __name__ == "__main__":
    main()
