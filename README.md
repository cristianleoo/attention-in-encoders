# Encoder Attention Explorer

An interactive visualization tool for exploring bidirectional self-attention mechanisms in Transformer models (BERT, RoBERTa, DeBERTa, and ModernBERT).

---

## 📄 Research Paper

This project is the official implementation for the paper:  
**"Survey of Attention Mechanisms in Encoder-Only Language Models"**  
[Read the paper on arXiv](https://arxiv.org/abs/XXXX.XXXXX)

---

## 🎨 Interactive Explorer

Probe bidirectional self-attention, visualize token-to-token flow, and benchmark hardware limits across multiple model architectures.

### Quickstart

To run both the backend (FastAPI) and frontend (Next.js) from the project root:

1.  **Install dependencies** (run once):
    ```bash
    npm run install:all
    ```
2.  **Start the application**:
    ```bash
    npm run dev
    ```
    - **Frontend:** `http://localhost:3000`
    - **Backend API:** `http://localhost:8000`

---

## 📂 Project Structure

```
attention-in-encoders/
├── app/
│   ├── backend/    FastAPI + HuggingFace (Inference Engine)
│   └── frontend/   Next.js (React Explorer UI)
├── docs/           System architecture & API reference
└── tests/          Automated backend & frontend tests
```

---

## 🛠 Supported Models

| Model | Attention Type | Context |
|---|---|---|
| `bert-base-uncased` | Standard Bidirectional MHA | 512 tokens |
| `roberta-base` | Standard Bidirectional MHA | 512 tokens |
| `microsoft/deberta-v3-base` | Disentangled (Content + Position) | 512 tokens |
| `answerdotai/ModernBERT-base` | Alternating Global / Local | 8,192 tokens |
| `BAAI/bge-m3` | Multi-Granularity Embedding | 8,192 tokens |
| `Alibaba-NLP/gte-modernbert-base` | GTE-ModernBERT Retrieval | 8,192 tokens |

---

## 🧪 Tests

```bash
npm run test
```

---

## ⚖️ License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.
