from __future__ import annotations

import argparse
import base64
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import run_cycle_comparison as comp  # noqa: E402


PRESENTATION_ROOT = PROJECT_ROOT / "data" / "outputs" / "presentation_process_figures"
SLIDES_ROOT = PROJECT_ROOT / "slides_Images"
CHARTS_ROOT = comp.OUTPUT_ROOT / "figures"

COPY_SUFFIXES = {
    ".csv",
    ".json",
    ".jpg",
    ".jpeg",
    ".md",
    ".npz",
    ".png",
    ".txt",
    ".webp",
}


DEMO_RUNS = [
    {
        "folder": "pair_1_main_diff_sdxl",
        "display": "Pair 1 main pipeline, SDXL diffusion",
        "pipeline": "main",
        "variant": "diff_sdxl",
        "input_variant": "diff_sdxl",
        "input_pair": "pair_1",
        "pair_slug": "slides_pair_1_main_diff_sdxl",
    },
    {
        "folder": "pair_2_main_diff_sdxl",
        "display": "Pair 2 main pipeline, SDXL diffusion",
        "pipeline": "main",
        "variant": "diff_sdxl",
        "input_variant": "diff_sdxl",
        "input_pair": "pair_2",
        "pair_slug": "slides_pair_2_main_diff_sdxl",
    },
    {
        "folder": "pair_3_face_pair_facedrag_fast_pti",
        "display": "Pair 3 Face-DragGAN pipeline, fast PTI",
        "pipeline": "face_draggan",
        "variant": "face_draggan_fast_pti",
        "input_variant": "face_draggan_fast_pti",
        "input_pair": "face_pair",
        "pair_slug": "slides_pair_3_face_draggan_fast_pti",
    },
]


def assert_project_child(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Refusing to operate outside project root: {resolved}")


def reset_dir(path: Path) -> None:
    assert_project_child(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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


def copy_file(src: Path, dst: Path) -> Path:
    src = src.resolve()
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def copy_selected_tree(src_dir: Path, dst_dir: Path) -> list[Path]:
    src_dir = src_dir.resolve()
    if not src_dir.exists():
        return []
    reset_dir(dst_dir)
    copied = []
    for src in sorted(src_dir.rglob("*")):
        if not src.is_file() or src.suffix.lower() not in COPY_SUFFIXES:
            continue
        dst = dst_dir / src.relative_to(src_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def decode_png(data: Any) -> bytes:
    if isinstance(data, list):
        data = "".join(data)
    return base64.b64decode(data)


def extract_notebook_images(notebook_path: Path, out_dir: Path) -> list[Path]:
    if not notebook_path.exists():
        return []
    reset_dir(out_dir)
    notebook = nbformat.read(notebook_path, as_version=4)
    extracted = []
    image_index = 0
    for cell_index, cell in enumerate(notebook.cells, start=1):
        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            data = output.get("data", {})
            if "image/png" not in data:
                continue
            image_index += 1
            dst = out_dir / f"cell_{cell_index:03d}_output_{output_index:02d}_image_{image_index:03d}.png"
            dst.write_bytes(decode_png(data["image/png"]))
            extracted.append(dst)
    return extracted


def input_paths(input_variant: str, input_pair: str) -> tuple[Path, Path]:
    input_dir = comp.RUN_INPUT_ROOT / input_variant / input_pair
    org = input_dir / f"{input_pair}_forward_org.png"
    rep = input_dir / f"{input_pair}_forward_rep.png"
    if not org.exists() or not rep.exists():
        raise FileNotFoundError(f"Missing demo inputs under {input_dir}")
    return org, rep


def active_session_manifest_path(pipeline: str) -> Path:
    if pipeline == "face_draggan":
        return PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "01_cutout" / "_active_session.json"
    return PROJECT_ROOT / "data" / "outputs" / "01_cutout" / "_active_session.json"


def write_active_session_for_demo(item: dict[str, str], org_path: Path, rep_path: Path) -> Path:
    """Notebook 01 intentionally gives uploaded-session manifests priority.

    For unattended slide generation, we therefore write the active manifest
    explicitly before each pair. This makes the run deterministic and prevents
    a stale interactive upload from silently replacing the requested inputs.
    """
    manifest_path = active_session_manifest_path(item["pipeline"])
    manifest = {
        "source_mode": "scripted_slides_demo",
        "pair_id": item["pair_slug"],
        "pair_name": comp.pair_name_from_paths(org_path, rep_path),
        "original_path": str(org_path.resolve()),
        "replacement_path": str(rep_path.resolve()),
    }
    write_json(manifest_path, manifest)
    return manifest_path


def clean_route_outputs(item: dict[str, str]) -> None:
    variant = comp.VARIANTS[item["variant"]]
    route = f"eval_{variant.name}_{item['pair_slug']}_forward"
    if item["pipeline"] == "face_draggan":
        route_dirs = [
            PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "05_diffusion_refine" / route,
            PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "06_postprocess" / route,
        ]
    else:
        route_dirs = [
            PROJECT_ROOT / "data" / "outputs" / "03_diffusion_refine" / route,
            PROJECT_ROOT / "data" / "outputs" / "04_postprocess" / route,
        ]
    route_dirs.append(comp.EXECUTED_NOTEBOOK_ROOT / variant.name / item["pair_slug"] / "forward")
    for route_dir in route_dirs:
        if route_dir.exists():
            assert_project_child(route_dir)
            shutil.rmtree(route_dir)


def run_demo(item: dict[str, str], timeout: int, rerun: bool) -> dict[str, Any]:
    variant = comp.VARIANTS[item["variant"]]
    org_path, rep_path = input_paths(item["input_variant"], item["input_pair"])
    runner = comp.run_face_draggan_pipeline_once if item["pipeline"] == "face_draggan" else comp.run_pipeline_once
    if rerun:
        manifest_path = write_active_session_for_demo(item, org_path, rep_path)
        clean_route_outputs(item)
        print(f"[slides] active manifest: {rel(manifest_path)}")
        return runner(
            org_path=org_path,
            rep_path=rep_path,
            pair_slug=item["pair_slug"],
            direction="forward",
            variant=variant,
            timeout=timeout,
        )

    route = f"eval_{variant.name}_{item['pair_slug']}_forward"
    if item["pipeline"] == "face_draggan":
        manifest = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "05_diffusion_refine" / route / "batch_manifest.json"
        post_manifest = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "06_postprocess" / route / "batch_manifest.json"
        cutout_manifest = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "01_cutout" / "_active_batch_manifest.json"
        pair_name = read_json(cutout_manifest)[0]["pair_name"]
        cutout_meta_path = PROJECT_ROOT / read_json(cutout_manifest)[0]["metadata_path"]
        warp_meta_path = PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "04_body_warp_after_true_draggan" / pair_name / "metadata.json"
    else:
        manifest = PROJECT_ROOT / "data" / "outputs" / "03_diffusion_refine" / route / "batch_manifest.json"
        post_manifest = PROJECT_ROOT / "data" / "outputs" / "04_postprocess" / route / "batch_manifest.json"
        cutout_manifest = PROJECT_ROOT / "data" / "outputs" / "01_cutout" / "_active_batch_manifest.json"
        pair_name = read_json(cutout_manifest)[0]["pair_name"]
        cutout_meta_path = PROJECT_ROOT / read_json(cutout_manifest)[0]["metadata_path"]
        warp_meta_path = PROJECT_ROOT / "data" / "outputs" / "02_pose_warp" / pair_name / "metadata.json"
    refine_meta_path = Path(read_json(manifest)[0]["run_config_path"])
    post_meta_path = Path(read_json(post_manifest)[0].get("metadata_path") or read_json(post_manifest)[0]["run_summary_path"])
    post_meta = read_json(post_meta_path)
    final_key = "postprocessed_final"
    return {
        "pair_name": pair_name,
        "route": route,
        "final_path": Path(post_meta["outputs"][final_key]),
        "final_stage": "postprocess",
        "cutout_meta_path": cutout_meta_path,
        "warp_meta_path": warp_meta_path,
        "refine_manifest_path": manifest,
        "post_manifest_path": post_manifest,
        "post_meta_path": post_meta_path,
        "refine_meta_path": refine_meta_path,
    }


def executed_notebook_paths(item: dict[str, str]) -> list[tuple[str, Path]]:
    variant = comp.VARIANTS[item["variant"]]
    log_dir = comp.EXECUTED_NOTEBOOK_ROOT / variant.name / item["pair_slug"] / "forward"
    if item["pipeline"] == "face_draggan":
        return [
            ("01_face_cutout", log_dir / "01_face_executed.ipynb"),
            ("02_face_inversion", log_dir / "02_face_inversion_executed.ipynb"),
            ("03_face_edit", log_dir / "03_face_edit_executed.ipynb"),
            ("04_face_body_warp", log_dir / "04_face_body_warp_executed.ipynb"),
            ("05_face_diffusion_refine", log_dir / "05_face_diffusion_executed.ipynb"),
            ("06_face_postprocess", log_dir / "06_face_postprocess_executed.ipynb"),
        ]
    return [
        ("01_cutout", log_dir / "01_executed.ipynb"),
        ("02_pose_warp", log_dir / "02_executed.ipynb"),
        ("03_diffusion_refine", log_dir / "03_executed.ipynb"),
        ("04_postprocess", log_dir / "04_executed.ipynb"),
    ]


def archive_demo(item: dict[str, str], result: dict[str, Any], root: Path) -> list[dict[str, str]]:
    demo_dir = root / item["folder"]
    reset_dir(demo_dir)
    rows = []

    metadata_dir = demo_dir / "00_run_metadata_and_final"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    final_copy = copy_file(Path(result["final_path"]), metadata_dir / "forward_final.png")
    run_info = {
        "display": item["display"],
        "variant": item["variant"],
        "pipeline": item["pipeline"],
        "pair_slug": item["pair_slug"],
        "pair_name": result["pair_name"],
        "route": result["route"],
        "source_final_path": rel(Path(result["final_path"])),
        "archived_final_path": rel(final_copy),
    }
    write_json(metadata_dir / "run_info.json", run_info)
    rows.append({"demo": item["folder"], "stage": "00_run_metadata_and_final", "path": rel(final_copy)})

    for key in ["cutout_meta_path", "warp_meta_path", "refine_manifest_path", "post_manifest_path", "post_meta_path"]:
        value = result.get(key)
        if value:
            src = Path(value)
            if src.exists():
                dst = copy_file(src, metadata_dir / f"{key}.json")
                rows.append({"demo": item["folder"], "stage": "00_run_metadata_and_final", "path": rel(dst)})

    for stage, notebook_path in executed_notebook_paths(item):
        extracted = extract_notebook_images(notebook_path, demo_dir / stage)
        rows.extend({"demo": item["folder"], "stage": stage, "path": rel(path)} for path in extracted)

    cutout_meta = Path(result["cutout_meta_path"])
    warp_meta = Path(result["warp_meta_path"])
    refine_route_dir = Path(result["refine_manifest_path"]).parent
    post_manifest = result.get("post_manifest_path")
    post_route_dir = Path(post_manifest).parent if post_manifest else None

    stage_sources = [
        ("saved_01_cutout_artifacts", cutout_meta.parent),
        ("saved_02_pose_warp_artifacts", warp_meta.parent),
        ("saved_03_diffusion_refine_artifacts", refine_route_dir),
    ]
    if post_route_dir is not None:
        stage_sources.append(("saved_04_postprocess_artifacts", post_route_dir))

    if item["pipeline"] == "face_draggan":
        pair_name = result["pair_name"]
        stage_sources.extend(
            [
                (
                    "saved_02_face_inversion_artifacts",
                    PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "02_true_draggan_face_inversion" / pair_name,
                ),
                (
                    "saved_03_face_edit_artifacts",
                    PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "03_true_draggan_face_edit" / pair_name,
                ),
                (
                    "saved_04_face_body_warp_artifacts",
                    PROJECT_ROOT / "data" / "outputs" / "face_draggan_pipeline" / "04_body_warp_after_true_draggan" / pair_name,
                ),
            ]
        )

    for stage, src_dir in stage_sources:
        copied = copy_selected_tree(src_dir, demo_dir / stage)
        rows.extend({"demo": item["folder"], "stage": stage, "path": rel(path)} for path in copied)
    return rows


def copy_charts(root: Path) -> list[dict[str, str]]:
    rows = []
    dst_dir = root / "00_comparison_charts"
    reset_dir(dst_dir)
    if not CHARTS_ROOT.exists():
        return rows
    for src in sorted(CHARTS_ROOT.iterdir()):
        if not src.is_file() or src.suffix.lower() not in COPY_SUFFIXES:
            continue
        dst = copy_file(src, dst_dir / src.name)
        rows.append({"demo": "00_comparison_charts", "stage": "charts", "path": rel(dst)})
    return rows


def write_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    manifest_path = root / "manifest.csv"
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["demo", "stage", "path"])
        writer.writeheader()
        writer.writerows(rows)
    readme = root / "README.md"
    readme.write_text(
        "# Slides Images\n\n"
        "This folder is generated by `scripts/build_slides_images.py`.\n\n"
        "Important: each demo pair is run and archived immediately, so shared notebook output folders "
        "such as `org_org__rep_rep` cannot be overwritten by the next pair before the slide images are copied.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and archive presentation process images without cross-pair overwrite.")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-rerun", action="store_true", help="Only rebuild folders from existing unique-route outputs.")
    args = parser.parse_args()

    reset_dir(PRESENTATION_ROOT)
    rows = copy_charts(PRESENTATION_ROOT)
    for item in DEMO_RUNS:
        print(f"\n[slides] {item['display']}")
        result = run_demo(item, timeout=args.timeout, rerun=not args.no_rerun)
        rows.extend(archive_demo(item, result, PRESENTATION_ROOT))
        print(f"[slides] archived {item['folder']} from route {result['route']}")

    write_manifest(PRESENTATION_ROOT, rows)

    if SLIDES_ROOT.exists():
        shutil.rmtree(SLIDES_ROOT)
    shutil.copytree(PRESENTATION_ROOT, SLIDES_ROOT)
    print(f"\nWrote {len(rows)} files to {rel(PRESENTATION_ROOT)}")
    print(f"Copied presentation folder to {rel(SLIDES_ROOT)}")


if __name__ == "__main__":
    main()
