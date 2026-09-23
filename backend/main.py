from __future__ import annotations

import copy
import io
import os
import re
from collections import Counter
from typing import Any

import nbformat
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from nbclient import NotebookClient
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv
from fastapi.responses import JSONResponse
from openai import OpenAI

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip()
MAX_NOTEBOOK_BYTES = 5 * 1024 * 1024
MAX_EXECUTED_CELLS = 8

app = FastAPI(title="NotebookPilot API", version="0.1.0")
api_key = os.getenv("OPENROUTER_API_KEY")
model_name = os.getenv("OPENROUTER_MODEL")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    notebook_path: str
    cells: list[dict]


class SelectedCell(BaseModel):
    index: int = Field(description="Zero-based notebook cell index")
    reason: str = Field(description="Short reason this cell is needed")


class SelectionPlan(BaseModel):
    selected: list[SelectedCell]
    plan: list[str] = Field(default_factory=list)


def _read_notebook(raw: bytes):
    if len(raw) > MAX_NOTEBOOK_BYTES:
        raise HTTPException(413, "Notebook is larger than 5 MB.")
    try:
        return nbformat.reads(raw.decode("utf-8"), as_version=4)
    except Exception as exc:
        raise HTTPException(400, f"Could not parse notebook: {exc}") from exc


def _cell_catalog(nb) -> list[dict[str, Any]]:
    catalog = []
    for i, cell in enumerate(nb.cells):
        source = cell.get("source", "")
        catalog.append(
            {
                "index": i,
                "type": cell.cell_type,
                "preview": source[:350],
                "source": source[:1800],
            }
        )
    return catalog


def _gemini_select(nb, task: str) -> SelectionPlan | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    model_name = os.getenv("OPENROUTER_MODEL", "openrouter/free")

    if not api_key:
        return None

    cells = _cell_catalog(nb)

    compact = "\n\n".join(
        f"CELL {c['index']} [{c['type']}]\n{c['source']}"
        for c in cells
    )

    prompt = f"""
You are selecting Jupyter notebook cells to execute for a user's task.

USER TASK:
{task}

NOTEBOOK:
{compact}

Choose the smallest useful subset of CODE cells that can accomplish the task.

Important rules:
- Preserve original execution order.
- Include prerequisite import/setup/data-loading cells when required.
- Do not select markdown cells.
- Prefer at most {MAX_EXECUTED_CELLS} code cells.
- Never invent cell indices.
- Give a very short reason for each selected cell.
- Give a short overall execution plan.

Return ONLY valid JSON in this exact format:

{{
  "selected": [
    {{
      "index": 0,
      "reason": "short reason"
    }}
  ],
  "plan": "short execution plan"
}}
"""

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Notebook Pilot. "
                        "Select the minimum notebook cells needed "
                        "to accomplish the user's task. "
                        "Return only valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if not answer:
            return None

        # Some models may still wrap JSON in ```json ... ```
        answer = answer.strip()

        if answer.startswith("```"):
            answer = answer.removeprefix("```json")
            answer = answer.removeprefix("```")
            answer = answer.removesuffix("```")
            answer = answer.strip()

        plan = SelectionPlan.model_validate_json(answer)

    except Exception as e:
        print("OpenRouter cell selection failed:", repr(e))
        return None

    # Validate that the model only selected real, non-empty code cells.
    valid_code = {
        i
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.source.strip()
    }

    seen = set()
    selected = []

    for item in sorted(plan.selected, key=lambda x: x.index):
        if item.index in valid_code and item.index not in seen:
            selected.append(item)
            seen.add(item.index)

    plan.selected = selected[:MAX_EXECUTED_CELLS]

    return plan if plan.selected else None


STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "into", "then", "show",
    "use", "using", "run", "cell", "cells", "notebook", "please", "get", "make",
    "what", "where", "when", "which", "report", "output", "result", "results",
}


def _tokens(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", text.lower())
        if t not in STOPWORDS
    }


def _heuristic_select(nb, task: str) -> SelectionPlan:
    task_tokens = _tokens(task)
    code_indices = [
        i for i, c in enumerate(nb.cells) if c.cell_type == "code" and c.source.strip()
    ]
    if not code_indices:
        raise HTTPException(400, "Notebook has no executable code cells.")

    scores: dict[int, float] = {}
    for i in code_indices:
        src_tokens = _tokens(nb.cells[i].source)
        overlap = len(task_tokens & src_tokens)
        score = float(overlap)
        lowered = nb.cells[i].source.lower()
        intent_words = ["fit", "predict", "plot", "train", "evaluate", "mean", "model"]
        for keyword in intent_words:
            if keyword in task.lower() and keyword in lowered:
                score += 0.35
        scores[i] = score

    ranked = sorted(code_indices, key=lambda i: (scores[i], i), reverse=True)
    core = [i for i in ranked if scores[i] > 0][:2]
    if not core:
        core = code_indices[-2:]

    # Include likely setup/import/data cells before the furthest selected cell.
    furthest = max(core)
    setup = []
    for i in code_indices:
        if i >= furthest:
            continue
        src = nb.cells[i].source.lower()
        if (
            src.lstrip().startswith(("import ", "from "))
            or "read_csv" in src
            or "load_" in src
            or "np.random" in src
            or "pd.dataframe" in src
        ):
            setup.append(i)

    chosen = sorted(set(setup + core))[:MAX_EXECUTED_CELLS]
    reasons = []
    for i in chosen:
        reason = "Prerequisite setup/data cell" if i in setup and i not in core else "Matches the requested task"
        reasons.append(SelectedCell(index=i, reason=reason))

    return SelectionPlan(
        selected=reasons,
        plan=[
            "Identify task-relevant code and its prerequisites",
            "Execute only that subset in original notebook order",
            "Return outputs and a transparent execution trace",
        ],
    )


def _select(nb, task: str) -> tuple[SelectionPlan, str]:
    try:
        plan = _gemini_select(nb, task)
        if plan:
            return plan, "gemini"
    except Exception as exc:
        print(f"Gemini selection failed; using fallback: {exc}")
    return _heuristic_select(nb, task), "heuristic"


def _serialize_output(output: Any) -> dict[str, Any]:
    output_type = output.get("output_type", "unknown")
    if output_type == "stream":
        return {"type": "stream", "text": output.get("text", "")}
    if output_type in {"execute_result", "display_data"}:
        data = output.get("data", {})
        if "text/plain" in data:
            return {"type": output_type, "text": str(data["text/plain"])}
        return {"type": output_type, "text": "[rich output produced]"}
    if output_type == "error":
        return {
            "type": "error",
            "text": f"{output.get('ename', 'Error')}: {output.get('evalue', '')}",
        }
    return {"type": output_type, "text": str(output)[:1200]}


def _execute_subset(nb, plan: SelectionPlan):
    chosen = [item.index for item in plan.selected]
    subset = nbformat.v4.new_notebook(metadata=copy.deepcopy(nb.metadata))
    subset.cells = [copy.deepcopy(nb.cells[i]) for i in chosen]

    kernel_name = (
        nb.metadata.get("kernelspec", {}).get("name")
        or os.getenv("JUPYTER_KERNEL", "python3")
    )
    client = NotebookClient(
        subset,
        timeout=30,
        kernel_name=kernel_name,
        allow_errors=True,
        store_widget_state=False,
    )
    executed = client.execute()

    trace = []
    for original_index, item, executed_cell in zip(chosen, plan.selected, executed.cells):
        outputs = [_serialize_output(out) for out in executed_cell.get("outputs", [])]
        trace.append(
            {
                "index": original_index,
                "reason": item.reason,
                "source": nb.cells[original_index].source[:1400],
                "outputs": outputs,
            }
        )
    return trace


@app.get("/api/health")
def health():
    return {"ok": True, "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))}


@app.post("/api/inspect")
async def inspect_notebook(file: UploadFile = File(...)):
    raw = await file.read()
    nb = _read_notebook(raw)
    return {
        "filename": file.filename,
        "cells": _cell_catalog(nb),
        "code_cells": sum(1 for c in nb.cells if c.cell_type == "code"),
    }


@app.post("/api/run")
async def run_agent(file: UploadFile = File(...), task: str = Form(...)):
    if not task.strip():
        raise HTTPException(400, "Please provide a task.")
    raw = await file.read()
    nb = _read_notebook(raw)
    plan, mode = _select(nb, task.strip())
    trace = _execute_subset(nb, plan)
    return {
        "task": task.strip(),
        "mode": mode,
        "plan": plan.plan,
        "selected_cells": [item.model_dump() for item in plan.selected],
        "trace": trace,
        "executed_count": len(trace),
        "total_cells": len(nb.cells),
    }

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     notebook_text = "\n\n".join(
#         f"Cell {i} ({cell.get('cell_type')}):\n{cell.get('source', '')}"
#         for i, cell in enumerate(req.cells)
#     )

#     full_prompt = f"""
# You are Notebook Pilot, an assistant helping a user understand
# and work with their Jupyter notebook.

# Notebook: {req.notebook_path}

# Notebook cells:
# {notebook_text}

# User request:
# {req.prompt}

# Answer based on the notebook.
# """

#     response = client.models.generate_content(
#         model="YOUR_CURRENT_GEMINI_MODEL",
#         contents=full_prompt,
#     )

#     return {
#         "response": response.text
#     }
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        notebook_text = "\n\n".join(
            f"Cell {i} ({cell.get('cell_type')}):\n"
            f"{cell.get('source', '')}"
            for i, cell in enumerate(req.cells)
        )

        full_prompt = f"""
Notebook: {req.notebook_path}

Notebook cells:
{notebook_text}

User request:
{req.prompt}

Answer based on the notebook.
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Notebook Pilot, an assistant helping "
                        "a user understand and work with Jupyter notebooks."
                    ),
                },
                {
                    "role": "user",
                    "content": full_prompt,
                },
            ],
        )

        answer = response.choices[0].message.content

        return {
            "response": answer
        }

    except Exception as e:
        print("CHAT ERROR:", repr(e))

        return JSONResponse(
            status_code=500,
            content={
                "error": type(e).__name__,
                "detail": str(e),
            },
        )