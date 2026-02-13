import torch
from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from utils.config import model_name
from settings import train_df_path, test_df_path
from utils.file_handler import read_csv
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

    def get_embedding(self, sequence, pooling="tokens"):
        input_ids = torch.tensor([self.tokenizer.encode(sequence).ids]).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]  # (1, L, 1024)

        if pooling == "tokens":
            embedding = last_hidden_state.squeeze(0)          # (L, 1024)
        elif pooling == "last":
            embedding = last_hidden_state[:, -1, :].squeeze(0) # (1024,)
        elif pooling == "mean":
            embedding = last_hidden_state.mean(dim=1).squeeze(0) # (1024,)
        else:
            raise ValueError("pooling must be 'tokens', 'last', or 'mean'")

        return embedding



train_df = read_csv(train_df_path)
extractor = ProGenEmbeddingExtractor(model_name=model_name)
print(train_df.columns)

for index, row in train_df[:10].iterrows():
    seq = row['original_sequence']
    print(len(seq))
    embedding = extractor.get_embedding(seq, pooling="tokens")
    print("Embedding shape:", embedding.shape)


