from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
import torch


class BERTEmbeddings:
    """
    Generate sentence embeddings based on a given method and model name.
    """

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "bert-base-uncased")
        self.private_model = kwargs.get("private_model", False)
        self.batch_size = kwargs.get("batch_size", 4)
        self.corpus = kwargs.get("corpus", None)
        self.with_gpu = kwargs.get("with_gpu", False)
        self.method = kwargs.get("method", "bert")
        self.hf_token = kwargs.get("hf_token", None)

        if self.private_model and self.hf_token is None:
            raise Exception("Please supply a token when using private models!")

        if self.hf_token:
            login(self.hf_token)

        if self.method == "bert":
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)

    def generate_embeddings(self):
        """

        Generate the embeddings with the given class object.
        :return: The embeddings from the model supplied.
        :rtype: PyTorch Tensor
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(
            self.corpus, padding=True, truncation=True, return_tensors="pt"
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        self.model = self.model.to(device)

        embeddings = []
        for i in range(0, len(self.corpus), self.batch_size):
            input_ids_batch = input_ids[i : i + self.batch_size]
            attention_mask_batch = attention_mask[i : i + self.batch_size]

            with torch.no_grad():
                input_ids_batch = input_ids_batch.to(device)
                attention_mask_batch = attention_mask_batch.to(device)

                outputs = model(input_ids_batch, attention_mask=attention_mask_batch)

            embeddings_batch = outputs.last_hidden_state[:, 0, :]
            embeddings_batch = embeddings_batch.cpu()

            embeddings.extend(embeddings_batch)

        embeddings = torch.stack(embeddings)
        return embeddings

