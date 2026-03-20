# Attention in Encoder-Only Models

Research paper + interactive visualization exploring bidirectional self-attention in BERT, RoBERTa, DeBERTa, and ModernBERT.

---

## Project Structure

```
attention-in-encoders/
├── paper/          LaTeX source for arXiv submission
│   ├── main.tex
│   └── references.bib
├── app/
│   ├── backend/    FastAPI + HuggingFace inference
│   └── frontend/   Next.js interactive UI
├── docs/           Architecture & API docs
└── tests/          Backend tests
```

---

## Quickstart

### 1 · Paper (LaTeX)

Requires a LaTeX distribution (e.g. TeX Live or MiKTeX):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex && pdflatex main.tex
```

### 2 · Backend

```bash
cd app/backend
source venv/bin/activate          # or venv\Scripts\activate on Windows
uvicorn main:app --reload --port 8000
```

API will be available at `http://localhost:8000`.

### 3 · Frontend

```bash
cd app/frontend
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## Supported Models

| Model | Attention Type | Context |
|---|---|---|
| `bert-base-uncased` | Standard Bidirectional MHA | 512 tokens |
| `roberta-base` | Standard Bidirectional MHA | 512 tokens |
| `microsoft/deberta-v3-base` | Disentangled (Content + Position) | 512 tokens |
| `answerdotai/ModernBERT-base` | Alternating Global / Local | 8 192 tokens |

---

## Tests

```bash
cd tests
python -m pytest backend/ -v
```

---

## Docs

See [`docs/architecture.md`](docs/architecture.md) for the full API reference and system architecture.