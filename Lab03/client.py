import tritonclient.http as httpclient
import numpy as np
import hydra
from scipy.special import softmax

from data import MyDataModule

PORT = "8000"

@hydra.main(config_path="./conf", config_name="config", version_base="1.2")
def run_triton_inderence(cfg):
    lables = ["unacceptable", "acceptable"]

    data = MyDataModule(model_name=cfg.model.name,
                        tokenizer_name=cfg.model.tokenizer,
                        batch_size=cfg.processing.batch_size,
                        max_length=cfg.processing.max_length)
    
    data.prepare_data()
    data.setup()

    input_batch = next(iter(data.train_dataloader()))

    #! Здесь могут быть проблемы с размерностями, исследовать внутр. структуру input_batch['input_ids']
    input_ids = httpclient.InferInput("input_ids", input_batch["input_ids"].shape, "INT32")
    attention_mask = httpclient.InferInput("attention_mask", input_batch["attention_mask"].shape, "INT32")

    triton_server_url = f"localhost:{PORT}"
    client = httpclient.InferenceServerClient(url=triton_server_url)

    #! также, проследить, правильную ли размерность мы здесь выдаем
    input_ids.set_data_from_numpy(input_batch["input_ids"][0].unsqueeze(0))
    attention_mask.set_data_from_numpy(input_batch["attention_mask"][0].unsqueeze(0))

    output = httpclient.InferRequestedOutput("logits")

    response = client.infer(
        model_name="hugging_face_model",
        inputs=[input_ids, attention_mask],
        outputs=[output]
    )

    logits = response.as_numpy("output")
    scores = softmax(logits)[0]
    predictions = []
    for score, label in zip(scores, lables):
        predictions.append({"label": label, "score": score})
    return predictions

if __name__ == '__main__':
    prediction = run_triton_inderence()
    print(prediction)