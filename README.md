# NotebookPilot

**An AI agent that finds and executes only the Jupyter cells relevant to a task.**

Instead of manually figuring out which cells need to be re-run in a long experimental notebook, upload an `.ipynb`, describe the outcome you want, and NotebookPilot:

1. parses the notebook structure,
2. uses Gemini to select the smallest useful cell path plus prerequisites,
3. executes that subset in a Jupyter kernel,
4. displays a transparent trace with selection reasons and outputs.

The app still works without an API key using a deterministic fallback planner, which makes the demo reliable.

## Why this exists

Experimental notebooks get messy: stale state, one-off setup cells, abandoned branches, and long dependency chains. The useful abstraction is not “run all” — it is **task-conditioned execution**.

## Stack

- **Frontend:** React + Vite
- **Backend:** FastAPI
- **AI:** Gemini via the `google-genai` SDK with structured JSON output
- **Notebook runtime:** `nbformat` + `nbclient`

## Run locally

### 1. Backend

```bash
cd backend
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: enable Gemini planning
# macOS/Linux
export GEMINI_API_KEY="your-key"
# Windows PowerShell: $env:GEMINI_API_KEY="your-key"

uvicorn main:app --reload --port 8000
```

### 2. Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the local Vite URL (normally `http://localhost:5173`).

### 3. Try the included demo

Upload `demo/demo_notebook.ipynb` and use:

> Fit a line to the synthetic data and report the learned coefficients.

The desired behavior is to execute the import/data/model-fit path while skipping the unrelated SVD experiment.

## API

`POST /api/run` accepts multipart form data:

- `file`: `.ipynb`
- `task`: natural-language task

It returns the planner mode, selected cells, reasons, execution plan, cell source, and outputs.

## Demo talking points

> NotebookPilot is a task-conditioned execution agent for Jupyter notebooks. It parses the real notebook graph, asks an LLM for the minimal relevant execution path, then runs only those cells in a fresh kernel and exposes the full trace. I built the fallback planner so the demo doesn't depend on an API call succeeding.

## Scope / safety

This is intentionally a local MVP. It executes notebook code, so only run notebooks you trust. A production version should execute in isolated containers with resource/network limits and add a dependency graph / static analysis pass before LLM planning.

## Obvious next features

- infer variable dependencies between cells using AST analysis,
- Docker/firecracker sandbox for execution,
- cache cell outputs and invalidate only affected dependencies,
- conversational follow-ups over a notebook session,
- VS Code / JupyterLab extension.
