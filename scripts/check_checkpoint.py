"""Offline tiny-model checkpoint test; not a trained model or performance benchmark."""

import importlib.metadata
import json
from pathlib import Path
import tempfile

import torch
from transformers import DistilBertConfig, DistilBertForSequenceClassification, Trainer, TrainingArguments


def main():
    torch.manual_seed(42)
    config = DistilBertConfig(
        vocab_size=64, max_position_embeddings=32, n_layers=1, n_heads=2,
        dim=16, hidden_dim=32, num_labels=2, dropout=0.0, attention_dropout=0.0,
    )
    model = DistilBertForSequenceClassification(config)
    examples = [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1], "labels": i % 2}
        for i in range(8)
    ]
    with tempfile.TemporaryDirectory() as directory:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=directory, use_cpu=True, max_steps=2,
                per_device_train_batch_size=4, per_device_eval_batch_size=4,
                eval_strategy="steps", eval_steps=1, save_strategy="steps", save_steps=1,
                load_best_model_at_end=True, metric_for_best_model="loss",
                report_to="none", disable_tqdm=True, dataloader_pin_memory=False,
                save_total_limit=1,
            ),
            train_dataset=examples, eval_dataset=examples,
        )
        trainer.train()
        reference, info = DistilBertForSequenceClassification.from_pretrained(
            trainer.state.best_model_checkpoint, output_loading_info=True
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            assert not info.get(key), (key, info[key])
        trainer.model.eval()
        reference.eval()
        inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]]), "attention_mask": torch.ones((1, 4), dtype=torch.long)}
        with torch.no_grad():
            before = trainer.model(**inputs).logits
            reloaded = reference(**inputs).logits
        assert torch.equal(before, reloaded), "Trainer's selected model differs from direct checkpoint reload."
        export = Path(directory) / "export"
        trainer.save_model(export)
        copied, info = DistilBertForSequenceClassification.from_pretrained(export, output_loading_info=True)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            assert not info.get(key), (key, info[key])
        copied.eval()
        with torch.no_grad():
            after = copied(**inputs).logits
        assert torch.equal(before, after), "Saved/reloaded logits differ."
        print(json.dumps({
            "check": "PASS: 2-step synthetic tiny-model checkpoint selection, save/reload and exact logit equality",
            "scope": "Offline plumbing smoke test only; not the original trained checkpoint or a sentiment evaluation.",
            "versions": {name: importlib.metadata.version(name) for name in ("transformers", "torch", "accelerate", "safetensors")},
            "max_logit_difference": float((before - after).abs().max()),
        }, indent=2))


if __name__ == "__main__":
    main()
