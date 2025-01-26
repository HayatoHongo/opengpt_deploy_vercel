# api/main.py
# import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

# ログの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# モデル設定を格納する設定クラス
class ModelConfig:
    batch_size = 16
    input_sequence_length = 32
    embedding_dim = 64
    hidden_dim = 256
    attention_head_count = 4
    layer_count = 4
    dropout_rate = 0.0
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

# データローダークラス
class DataLoader:
    def __init__(self, text, config):
        self.config = config
        chars = sorted(list(set(text)))
        self.ctoi = {char: index for index, char in enumerate(chars)}
        self.itoc = {index: char for index, char in enumerate(chars)}
        self.vocab_size = len(chars)
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.train_data, self.val_data = self.split_data()

    def encode(self, text):
        return [self.ctoi[c] for c in text]

    def decode(self, indices):
        return ''.join([self.itoc.get(i, '') for i in indices])

    def split_data(self):
        n = int(0.9 * len(self.data))
        return self.data[:n], self.data[n:]

    def get_batch(self, split):
        data = self.train_data if split == 'train' else self.val_data
        if len(data) <= self.config.input_sequence_length:
            logging.error(f"Data length {len(data)} is too small for input_sequence_length {self.config.input_sequence_length}.")
            raise ValueError("Data length is smaller than input_sequence_length.")
        start_indices = torch.randint(len(data) - self.config.input_sequence_length, (self.config.batch_size,))

        input_sequences = torch.stack([
            data[start_index:start_index + self.config.input_sequence_length]
            for start_index in start_indices
        ])

        target_sequences = torch.stack([
            data[start_index + 1:start_index + self.config.input_sequence_length + 1]
            for start_index in start_indices
        ])

        return input_sequences.to(self.config.device_type), target_sequences.to(self.config.device_type)

# 位置埋め込みモジュール
class PositionEmbedding(nn.Module):
    def __init__(self, input_sequence_length, embedding_dim):
        super().__init__()
        self.position_embedding_layer = nn.Embedding(input_sequence_length, embedding_dim)

    def forward(self, input_indices):
        sequence_length = input_indices.shape[1]
        position_indices = torch.arange(sequence_length, device=input_indices.device)

        # ログ追加
        logging.debug(f"Sequence length: {sequence_length}")
        logging.debug(f"Position indices: {position_indices}")

        if sequence_length > self.position_embedding_layer.num_embeddings:
            logging.error(f"Sequence length {sequence_length} exceeds position_embedding_layer.num_embeddings {self.position_embedding_layer.num_embeddings}.")
            raise IndexError("Sequence length exceeds position embedding layer's capacity.")

        position_embeddings = self.position_embedding_layer(position_indices)
        return position_embeddings

# トークン埋め込みと位置埋め込みを統合したモジュール
class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim, input_sequence_length):
        super().__init__()
        self.token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding_module = PositionEmbedding(input_sequence_length, embedding_dim)

    def forward(self, input_indices):
        token_embeddings = self.token_embedding_layer(input_indices)
        try:
            position_embeddings = self.position_embedding_module(input_indices)
        except IndexError as e:
            logging.error(f"Error in position_embedding_module: {e}")
            logging.error(f"Input indices shape: {input_indices.shape}")
            logging.error(f"Maximum allowed sequence length: {self.position_embedding_module.position_embedding_layer.num_embeddings}")
            raise e
        embeddings = token_embeddings + position_embeddings
        return embeddings

# 単一のアテンションヘッド
class AttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dim, input_sequence_length, dropout_rate):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(input_sequence_length, input_sequence_length)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embeded_input_tensor):
        batch_size, sequence_length, embedding_dim = embeded_input_tensor.shape
        key_tensor = self.key(embeded_input_tensor)
        query_tensor = self.query(embeded_input_tensor)

        attention_weights = query_tensor @ key_tensor.transpose(-2, -1) * embedding_dim**-0.5
        attention_weights = attention_weights.masked_fill(self.tril[:sequence_length, :sequence_length] == 0, float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)

        value_tensor = self.value(embeded_input_tensor)
        return attention_weights @ value_tensor

# マルチヘッドアテンション
class MultiHeadAttention(nn.Module):
    def __init__(self, attention_head_count, head_size, embedding_dim, input_sequence_length, dropout_rate):
        super().__init__()
        self.heads = nn.ModuleList([
            AttentionHead(head_size, embedding_dim, input_sequence_length, dropout_rate)
            for _ in range(attention_head_count)
        ])
        self.proj = nn.Linear(attention_head_count * head_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        return self.dropout(self.proj(out))

# フィードフォワードネットワーク
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, attention_output_tensor):
        return self.net(attention_output_tensor)

# トランスフォーマーブロック
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, attention_head_count, input_sequence_length, dropout_rate):
        super().__init__()
        head_size = embedding_dim // attention_head_count
        self.self_attention = MultiHeadAttention(attention_head_count, head_size, embedding_dim, input_sequence_length, dropout_rate)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout_rate)
        self.layer_norm_1 = nn.LayerNorm(embedding_dim)
        self.layer_norm_2 = nn.LayerNorm(embedding_dim)

    def forward(self, embeded_input_tensor):
        attention_output_tensor = embeded_input_tensor + self.self_attention(self.layer_norm_1(embeded_input_tensor))
        final_output_tensor = attention_output_tensor + self.feed_forward(self.layer_norm_2(attention_output_tensor))
        return final_output_tensor

# 語彙ロジット層
class VocabularyLogits(nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        super().__init__()
        self.final_layer_norm = nn.LayerNorm(embedding_dim)
        self.vocab_projection = nn.Linear(embedding_dim, vocab_size)

    def forward(self, transformer_blocks_output_tensor):
        normalized_tensor = self.final_layer_norm(transformer_blocks_output_tensor)
        vocab_logits = self.vocab_projection(normalized_tensor)
        return vocab_logits

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, config, vocab_size):
        super().__init__()
        self.config = config  # 修正箇所: configを保存
        self.embedding = EmbeddingModule(vocab_size, config.embedding_dim, config.input_sequence_length)
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.embedding_dim, config.hidden_dim, config.attention_head_count, config.input_sequence_length, config.dropout_rate)
            for _ in range(config.layer_count)
        ])
        self.head = VocabularyLogits(config.embedding_dim, vocab_size)

    def generate(self, input_indices, max_new_tokens):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # シーケンス長を制限
                input_conditioned = input_indices[:, -self.config.input_sequence_length:]
                logging.debug(f"Input conditioned shape: {input_conditioned.shape}")

                # ロジット計算
                logits = self.forward(input_conditioned)
                logits = logits[:, -1, :]  # 最後のトークンのログit
                probs = torch.softmax(logits, dim=-1)

                # 次のトークンをサンプリング
                next_token = torch.multinomial(probs, num_samples=1)

                # トークン列を更新
                input_indices = torch.cat((input_indices, next_token), dim=1)

                # ログ追加
                logging.debug(f"Generated token: {next_token.item()}")
                logging.debug(f"Updated input_indices shape: {input_indices.shape}")

        return input_indices

    def forward(self, input_indices, target_indices=None):
        embedded_input = self.embedding(input_indices)
        transformed_output = self.blocks(embedded_input)
        vocab_logits = self.head(transformed_output)

        if target_indices is None:
            return vocab_logits

        batch_size, sequence_length, vocab_size = vocab_logits.shape
        vocab_logits = vocab_logits.view(batch_size * sequence_length, vocab_size)
        target_indices = target_indices.view(batch_size * sequence_length)
        loss = nn.CrossEntropyLoss()(vocab_logits, target_indices)
        return vocab_logits, loss

   
# モデルの絶対パスを環境変数から取得
MODEL_PATH = os.getenv('MODEL_PATH')

if MODEL_PATH is None:
    raise ValueError("環境変数 MODEL_PATH が設定されていません。")

# モデルのロード関数
def load_model(config, vocab_size, model_path=MODEL_PATH):
    model = BigramLanguageModel(config, vocab_size).to(config.device_type)
    model.load_state_dict(torch.load(model_path, map_location=config.device_type))
    model.eval()
    return model

# APIリクエストのモデル
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 500

class GenerateResponse(BaseModel):
    generated_text: str

# FastAPIのインスタンス作成
app = FastAPI(title="Bigram Language Model API", description="Generate text using a trained Bigram Language Model.", version="1.0.0")

# グローバル変数としてモデルとデータローダーを保持
model = None
data_loader = None
config = None

# アプリケーション起動時にモデルとデータローダーをロード
@app.on_event("startup")
def startup_event():
    global model, data_loader, config
    try:
        # 設定のロード
        config = ModelConfig()
        logging.info("Loaded ModelConfig.")

        # Load data
        # input.txt をコンテナ内の相対パスで指定
        with open('text2text/answer/input.txt', 'r', encoding='utf-8') as f:
            text_data = f.read()
        data_loader = DataLoader(text_data, config)
        logging.info("Initialized DataLoader.")

        # モデルのロード（絶対パスを使用）
        model = load_model(config, data_loader.vocab_size, MODEL_PATH)
        logging.info("Loaded model and weights.")

    except Exception as e:
        logging.error(f"Failed to load model: {e}", exc_info=True)
        raise e

# テキスト生成エンドポイント
@app.post("/generate", response_model=GenerateResponse)
def generate_text(request: GenerateRequest):
    try:
        prompt_text = request.prompt
        max_new_tokens = request.max_new_tokens

        logging.info(f"Received prompt: '{prompt_text}' with max_new_tokens={max_new_tokens}")

        # プロンプトをエンコードし、バッチ次元を追加
        initial_context = torch.tensor(data_loader.encode(prompt_text), dtype=torch.long, device=config.device_type)
        initial_context_unsqueeze = initial_context.unsqueeze(0)

        logging.debug(f"Initial context tensor: {initial_context_unsqueeze}")

        # 生成の開始
        generated_sequence = model.generate(initial_context_unsqueeze, max_new_tokens=max_new_tokens)
        logging.info("Completed text generation.")

        # 生成されたトークン列をデコードしてテキストとして出力
        generated_text = data_loader.decode(generated_sequence[0].tolist())

        return GenerateResponse(generated_text=generated_text)

    except Exception as e:
        logging.error(f"Error during text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ルートエンドポイント
@app.get("/")
def read_root():
    return {"message": "Welcome to the Bigram Language Model API! Use the /generate endpoint to generate text."}