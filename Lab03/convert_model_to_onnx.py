import torch
import hydra
import os

from data import MyDataModule
from model import MyModel

@hydra.main(config_path="./conf", config_name="config", version_base="1.2")
def convert_model(cfg):
    root_dir = hydra.utils.get_original_cwd()
    model_path = f"{root_dir}/checkpoints/best-checkpoint.ckpt"

    model = MyModel.load_from_checkpoint(model_path)
    model.eval()
    model.to("cpu")

    data = MyDataModule(model_name=cfg.model.name,
                        tokenizer_name=cfg.model.tokenizer,
                        batch_size=cfg.processing.batch_size,
                        max_length=cfg.processing.max_length)
    
    data.prepare_data()
    data.setup()

    input_batch = next(iter(data.train_dataloader()))
    
    input_sample = {
        "input_ids": input_batch["input_ids"][0].unsqueeze(0),
        "attention_mask": input_batch["attention_mask"][0].unsqueeze(0),
    }

    # Export the model
    torch.onnx.export(
        model,  # model being run
        (
            input_sample["input_ids"].to("cpu"),
            input_sample["attention_mask"].to("cpu"),
        ),  # model input (or a tuple for multiple inputs)
        f"{root_dir}/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,
        opset_version=14,
        input_names=["input_ids", "attention_mask"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={
            "input_ids": {0: "batch_size"},  # variable length axes
            "attention_mask": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

if __name__ == "__main__":
    convert_model()