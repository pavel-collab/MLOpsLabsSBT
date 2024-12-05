import tritonclient.http as httpclient
import numpy as np
import hydra
from scipy.special import softmax

from data import MyDataModule

PORT = "8000"

@hydra.main(config_path="./conf", config_name="config", version_base="1.2")
def run_triton_inderence(cfg):
    data = MyDataModule(model_name=cfg.model.name,
                        tokenizer_name=cfg.model.tokenizer,
                        batch_size=cfg.processing.batch_size,
                        max_length=cfg.processing.max_length)
    
    data.prepare_data()
    data.setup()

    lables = data.idx2label
    if lables == None:
        print("[Error] there are no labels to idx in data")
        return
    
    random_sample_sentence = data.sample_random_item()

    decoded_sentence = data.tokenizer.decode(random_sample_sentence['input_ids'], skip_special_tokens=True)
    true_label_idx = random_sample_sentence['label']
    print(f"true label is:\n\t{lables[true_label_idx]}")
    print(f"sentence is:\n\t{decoded_sentence}")

    d_input_ids = random_sample_sentence['input_ids'].reshape(1, -1)
    d_attention_mask = random_sample_sentence['attention_mask'].reshape(1, -1)

    input_ids = httpclient.InferInput("input_ids", d_input_ids.shape, "INT64")
    attention_mask = httpclient.InferInput("attention_mask", d_attention_mask.shape, "INT64")

    triton_server_url = f"localhost:{PORT}"
    client = httpclient.InferenceServerClient(url=triton_server_url)

    input_ids.set_data_from_numpy(np.array(d_input_ids).astype('int64'))
    attention_mask.set_data_from_numpy(np.array(d_attention_mask).astype('int64'))

    output = httpclient.InferRequestedOutput("output")

    response = client.infer(
        model_name="onnx-bert-uncased",
        inputs=[input_ids, attention_mask],
        outputs=[output]
    )

    logits = response.as_numpy("output")
    scores = softmax(logits)[0]
    predictions = {}
    for score, label in zip(scores, lables):
        predictions[lables[label]] = score

    predicted_label = max(predictions, key=predictions.get)
    print(f"The model output is:\n\t{predictions}\nThe predicted label is:\n\t{predicted_label}")

if __name__ == '__main__':
    run_triton_inderence()
