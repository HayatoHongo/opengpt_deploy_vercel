# ベースイメージとしてPython 3.9.6-slimを使用
FROM python:3.9.6-slim

# システムパッケージのインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルをコピー
COPY requirements.txt .

# Pythonパッケージのインストール
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 環境変数の設定（必要に応じて）
ENV MODEL_PATH=/app/bigram_language_model.pth

# ポート8080を開放
EXPOSE 8080

# `numpy` のインストール確認
RUN python -c "import numpy; print('Numpy version:', numpy.__version__)"

# サーバーを起動
CMD ["uvicorn", "text2text.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
