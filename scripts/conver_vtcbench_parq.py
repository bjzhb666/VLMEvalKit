import base64
import json
import os.path as osp
import sys

from datasets import load_dataset
from tqdm import tqdm

VTCBENCH = "MLLM-CL/VTCBench"
# or use your own local copy of the dataset


__doct__ = f"""
usage: python {sys.argv[0]} <parquet_path1> <parquet_path2> ...
Convert {VTCBENCH} to VLMEvalkit-compatible TSV format.
"""


def encode_image_bytes_to_base64(image_bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode()


def convert_to_tsv(parquet_path: str) -> None:
    tsv_path = parquet_path.replace(".parquet", ".tsv")
    if osp.exists(tsv_path):
        print(f"TSV file {tsv_path} already exists. Skipping conversion.")
        return

    category = f"{osp.basename(parquet_path).split('.')[0]}"

    def _gen_fields(example: dict, idx: int) -> dict:
        # function body for dataset.map()

        images: list[dict[str, bytes]] = example["images"]
        problem: str = example["problem"]
        answers: list[str] = example["answers"]
        b64_imgs: list[str] = [
            encode_image_bytes_to_base64(img["bytes"]) for img in images
        ]
        return {
            "index": f"vtcbenchs_{idx}",
            "question": problem,
            "image": b64_imgs[0] if len(b64_imgs) == 1 else b64_imgs,
            "answer": json.dumps(answers),
            "category": category,
        }

    dataset = load_dataset("parquet", data_files=parquet_path)["train"]
    old_columns = dataset.column_names
    dataset = dataset.map(
        _gen_fields,
        with_indices=True,
        num_proc=8,
        remove_columns=old_columns,
    )
    dataset.to_csv(tsv_path, sep="\t", index=False)
    # sample = dataset[0]
    # for k, v in sample.items():
    #     print(f"{k}: {v}" if k != "image" else f"{k}: <{len(v)} characters>")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
    else:
        for parquet_path in tqdm(sys.argv[1:]):
            convert_to_tsv(parquet_path)
