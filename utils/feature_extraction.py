import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from utils.config import model_name

class ProGenEmbeddingExtractor:
    def __init__(self, model_name="hugohrban/progen2-small", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True
        ).to(self.device)
        
        self.tokenizer = Tokenizer.from_pretrained(model_name)
        self.tokenizer.no_padding()
        
        self.model.eval()

    def get_embedding(self, sequence, pooling="last"):
        """
        sequence: str
        pooling: "last" or "mean"
        returns: torch tensor (hidden_size,)
        """
        
        input_ids = torch.tensor(
            [self.tokenizer.encode(sequence).ids]
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True
            )

        last_hidden_state = outputs.hidden_states[-1]  

        if pooling == "last":
            embedding = last_hidden_state[:, -1, :] 
        elif pooling == "mean":
            embedding = last_hidden_state.mean(dim=1)  
        else:
            raise ValueError("pooling must be 'last' or 'mean'")

        return embedding.squeeze(0)  

extractor = ProGenEmbeddingExtractor(model_name=model_name)

seq = "1MEVVIVTGMSGAGK"

embedding = extractor.get_embedding(seq, pooling="last")

print("Embedding shape:", embedding.shape)
