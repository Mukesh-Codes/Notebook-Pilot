import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowRight,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  FileCode2,
  Play,
  Sparkles,
  UploadCloud,
  Zap,
} from "lucide-react";
import "./styles.css";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function Badge({ children, tone = "default" }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

function App() {
  const [file, setFile] = useState(null);
  const [task, setTask] = useState("Fit a line to the synthetic data and report the learned coefficients.");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [openCells, setOpenCells] = useState({});
  const [dragging, setDragging] = useState(false);

  const selectedSet = useMemo(
    () => new Set(result?.selected_cells?.map((x) => x.index) || []),
    [result]
  );

  function chooseFile(nextFile) {
    if (!nextFile) return;
    if (!nextFile.name.endsWith(".ipynb")) {
      setError("Please choose a .ipynb notebook.");
      return;
    }
    setFile(nextFile);
    setResult(null);
    setError("");
  }

  async function run() {
    if (!file) {
      setError("Upload a notebook first.");
      return;
    }
    if (!task.trim()) {
      setError("Describe what you want the notebook to do.");
      return;
    }
    setRunning(true);
    setResult(null);
    setError("");
    const body = new FormData();
    body.append("file", file);
    body.append("task", task);
    try {
      const response = await fetch(`${API}/api/run`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Agent run failed.");
      setResult(data);
    } catch (e) {
      setError(e.message || "Could not reach the backend.");
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="logo"><Zap size={18}/></div>NotebookPilot</div>
        <div className="top-actions"><Badge>local execution</Badge><a href="#how">How it works</a></div>
      </header>

      <main>
        <section className="hero">
          <Badge tone="accent"><Sparkles size={13}/> AI-native notebook tooling</Badge>
          <h1>Stop hunting for the <span>right cell.</span></h1>
          <p>Tell NotebookPilot what you need. It finds the relevant Jupyter cells, runs only the useful path, and shows you exactly what happened.</p>
        </section>

        <section className="workspace">
          <div className="panel input-panel">
            <div className="panel-title"><span>1</span> Add a notebook</div>
            <label
              className={`dropzone ${dragging ? "dragging" : ""}`}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={(e) => { e.preventDefault(); setDragging(false); chooseFile(e.dataTransfer.files[0]); }}
            >
              <input type="file" accept=".ipynb" onChange={(e) => chooseFile(e.target.files[0])} />
              <div className="upload-icon"><UploadCloud size={24}/></div>
              {file ? (
                <><strong>{file.name}</strong><small>{(file.size / 1024).toFixed(1)} KB · ready</small></>
              ) : (
                <><strong>Drop your .ipynb here</strong><small>or click to browse · up to 5 MB</small></>
              )}
            </label>

            <div className="panel-title second"><span>2</span> What do you want done?</div>
            <textarea value={task} onChange={(e) => setTask(e.target.value)} placeholder="e.g. Train the model and show me validation accuracy" />
            <div className="chips">
              {["Plot the loss curve", "Run evaluation only", "Compute summary statistics"].map((x) => (
                <button key={x} onClick={() => setTask(x)}>{x}</button>
              ))}
            </div>

            {error && <div className="error">{error}</div>}
            <button className="run-button" disabled={running} onClick={run}>
              {running ? <><span className="spinner"/>Planning execution…</> : <><Play size={17} fill="currentColor"/>Run relevant cells<ArrowRight size={17}/></>}
            </button>
            <div className="safety">Runs notebook code on your local machine. Use notebooks you trust.</div>
          </div>

          <div className={`panel result-panel ${!result ? "empty" : ""}`}>
            {!result ? (
              <div className="empty-state">
                <div className="agent-orb"><Bot size={32}/></div>
                <h3>Your execution trace appears here</h3>
                <p>The agent will select the smallest useful path through your notebook and explain each choice.</p>
                <div className="mini-flow"><span>task</span><ChevronRight/><span>plan</span><ChevronRight/><span>cells</span><ChevronRight/><span>output</span></div>
              </div>
            ) : (
              <div className="results">
                <div className="result-header">
                  <div><div className="eyebrow">AGENT RUN COMPLETE</div><h2>{result.executed_count} of {result.total_cells} cells executed</h2></div>
                  <Badge tone={result.mode === "gemini" ? "accent" : "default"}>{result.mode === "gemini" ? "Gemini planner" : "Demo planner"}</Badge>
                </div>

                <div className="plan-box">
                  <div className="plan-heading"><Sparkles size={16}/>Execution plan</div>
                  {result.plan.map((step, i) => <div className="plan-step" key={i}><span>{i+1}</span>{step}</div>)}
                </div>

                <div className="trace-heading">Selected path</div>
                <div className="trace-list">
                  {result.trace.map((cell) => {
                    const open = !!openCells[cell.index];
                    return (
                      <div className="trace-card" key={cell.index}>
                        <button className="trace-top" onClick={() => setOpenCells((s) => ({...s, [cell.index]: !open}))}>
                          <div className="cell-num">[{cell.index}]</div>
                          <div className="trace-copy"><strong>Cell {cell.index}</strong><span>{cell.reason}</span></div>
                          <CheckCircle2 className="check" size={18}/>
                          {open ? <ChevronDown size={17}/> : <ChevronRight size={17}/>}
                        </button>
                        {open && <pre className="source">{cell.source}</pre>}
                        {cell.outputs?.length > 0 && (
                          <div className="outputs">
                            {cell.outputs.map((out, i) => <pre className={out.type === "error" ? "output error-output" : "output"} key={i}>{out.text}</pre>)}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="how" id="how">
          <div><FileCode2/><strong>Notebook-aware</strong><span>Parses real .ipynb structure instead of treating notebooks as flat text.</span></div>
          <div><Sparkles/><strong>Semantic planning</strong><span>Gemini selects task-relevant cells and prerequisite setup.</span></div>
          <div><Zap/><strong>Selective execution</strong><span>Runs only the chosen path in a Jupyter kernel and surfaces outputs.</span></div>
        </section>
      </main>

      <footer>NotebookPilot · a tiny AI agent for messy notebooks</footer>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
