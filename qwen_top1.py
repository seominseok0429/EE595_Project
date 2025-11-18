import os
import glob
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
from utils import classify_image, build_prompt


def moe_use_last4_forward(self, hidden_states):

    batch_size = hidden_states.shape[0]
    hidden_states = hidden_states.reshape(-1, self.hidden_size)

    router_logits = self.gate(hidden_states)
    probs = torch.softmax(router_logits, dim=-1, dtype=torch.float)

    topk_w, topk_i = torch.topk(probs, self.top_k, dim=-1)

    selected_w = topk_w[:, [1]]
    selected_i = topk_i[:, [1]]

    selected_w = selected_w / selected_w.sum(dim=-1, keepdim=True)
    selected_w = selected_w.to(router_logits.dtype)

    router_weights = torch.zeros_like(router_logits)
    router_weights.scatter_(1, selected_i, selected_w)

    hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
    output = self.experts(hidden_states, router_weights, selected_i)
    return output


def patch_last_moe_layer(model):
    moe_layers = []

    for layer in model.model.language_model.layers:
        if isinstance(layer.mlp, Qwen3VLMoeTextSparseMoeBlock):
            moe_layers.append(layer.mlp)

    last_moe = moe_layers[-1]  #
    last_moe.forward = moe_use_last4_forward.__get__(last_moe, last_moe.__class__)

    print(f"[OK] Patched ONLY the LAST MoE block (index: {len(moe_layers)-1})")


if __name__ == "__main__":
    val_dir = "/workspace/SSD_4T/EE_project/dataset/tiny-imagenet-200/val"
    image_paths = glob.glob(f"{val_dir}/*/images/*")
    print("Image path :", len(image_paths))

    classes = sorted(list(set([p.split("/")[-3] for p in image_paths])))

    model_name = "Qwen/Qwen3-VL-30B-A3B-Instruct"

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto"
    )

    patch_last_moe_layer(model)

    processor = AutoProcessor.from_pretrained(model_name)
    correct = 0
    total = len(image_paths)

    print("Tiny-ImageNet-200 Classification \n")

    for i, img_path in enumerate(tqdm(image_paths), start=1):
        gt = img_path.split("/")[-3]
        pred = classify_image(img_path, classes, processor, model)

        if pred == gt:
            correct += 1

        running_acc = correct / i * 100
        print(f"[{i}/{total}] [GT] {gt} | [Pred] {pred} | Running ACC: {running_acc:.2f}%")

    acc = correct / total * 100
    print("\n====================================")
    print(f"FINAL ACCURACY: {acc:.2f}% ({correct}/{total})")
    print("====================================\n")
