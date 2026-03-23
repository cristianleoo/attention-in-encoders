"use client";

import React, { useState, useMemo, useCallback } from "react";

// ── Constants ──────────────────────────────────────────────────────
const API = "http://localhost:8005";

const MODELS = [
  { id: "bert-base-uncased",           label: "BERT-base",       desc: "12L · 12H · 110M · 512 ctx",   maxLen: 512 },
  { id: "roberta-base",                label: "RoBERTa-base",    desc: "12L · 12H · 125M · 512 ctx",   maxLen: 512 },
  { id: "microsoft/deberta-v3-base",   label: "DeBERTa v3",      desc: "12L · 12H · Disentangled",     maxLen: 2048 },
  { id: "google/bigbird-roberta-base", label: "BigBird",         desc: "Sparse · Block Sparse · 4k",   maxLen: 4096 },
  { id: "answerdotai/ModernBERT-base", label: "ModernBERT",      desc: "22L · 12H · Alternating · 8k", maxLen: 8192 },
  { id: "BAAI/bge-m3",                 label: "BGE-M3",          desc: "Dense Retrieval · 8192 ctx",   maxLen: 8192 },
  { id: "Alibaba-NLP/gte-modernbert-base", label: "GTE-ModernBERT", desc: "Retrieval · ModernBERT arch", maxLen: 8192 },
];

const SEQ_LENGTHS = [64, 128, 256, 512, 1024, 2048];

// ── Types ──────────────────────────────────────────────────────────
type AttData = { tokens: string[]; attention: number[][][][] };
type LimitRow = { seq_len: number; latency_ms: number | null; memory_mb: number | null; status: string };
type LimitsData = { model: string; results: LimitRow[] };
type Tab = "attention" | "limits" | "flow" | "tokenization" | "leaderboard";

// ── Color helpers ─────────────────────────────────────────────────
function attnColor(v: number, boost: boolean = false): string {
  const scaled = boost && v > 1e-9 ? Math.pow(v, 0.35) : Math.max(0, Math.min(1, v));
  const r = Math.round(20  + 220 * scaled);
  const g = Math.round(8   + 30  * (1 - scaled));
  const b = Math.round(70  + 130 * scaled);
  return `rgb(${r},${g},${b})`;
}

function flowColor(v: number): string {
  // vibrant teal for outgoing flow
  const a = Math.max(0.08, v);
  return `rgba(56, 189, 248, ${a})`;
}

function entropy(row: number[]): number {
  return row.reduce((s, p) => s - (p > 1e-9 ? p * Math.log(p) : 0), 0);
}

// ── Fragmentation Analysis ────────────────────────────────────────
function getFragmentation(tokens: string[]): { wordCount: number; fragRatio: number; fragmentedWords: string[] } {
  let wordCount = 0;
  let subTokens = 0;
  const fragmentedWords: string[] = [];
  let currentWord = "";
  let isFragmented = false;

  tokens.forEach(t => {
    // Standard BERT/RoBERTa/ModernBERT markers
    const isSub = t.startsWith("##") || (!t.startsWith("Ġ") && !t.startsWith(" ") && wordCount > 0 && !["[CLS]","[SEP]","<s>","</s>","<pad>","[MASK]"].includes(t));
    
    if (isSub) {
      subTokens++;
      if (!isFragmented) {
        fragmentedWords.push(currentWord + t.replace("##", ""));
        isFragmented = true;
      }
    } else {
      wordCount++;
      currentWord = t.replace("Ġ", "").replace(" ", "");
      isFragmented = false;
    }
  });

  return { 
    wordCount: Math.max(1, wordCount), 
    fragRatio: (tokens.length / Math.max(1, wordCount)),
    fragmentedWords 
  };
}

function TokenHealth({ tokens }: { tokens: string[] }) {
  const { wordCount, fragRatio, fragmentedWords } = getFragmentation(tokens);
  const isBad = fragRatio > 1.4;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em" }}>TOKENIZATION HEALTH</div>
      <div style={{ display: "flex", gap: 8, alignItems: "baseline" }}>
        <span style={{ fontSize: 24, fontWeight: 800, color: isBad ? "#f87171" : "#34d399" }}>
          {fragRatio.toFixed(2)}
        </span>
        <span style={{ fontSize: 11, color: "var(--muted)" }}>tokens/word</span>
      </div>
      <div style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.4 }}>
        {isBad 
          ? "⚠️ High fragmentation. Attention may be diluted by intra-word reconstruction." 
          : "✅ Healthy density. Most words are represented by single semantic units."}
      </div>
      {fragmentedWords.length > 0 && (
        <div style={{ marginTop: 4 }}>
          <div style={{ fontSize: 9, color: "var(--muted)", marginBottom: 4 }}>SPLIT WORDS:</div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
            {fragmentedWords.slice(0, 5).map((w, i) => (
              <span key={i} className="chip" style={{ fontSize: 9, padding: "2px 6px" }}>{w}</span>
            ))}
            {fragmentedWords.length > 5 && <span style={{ fontSize: 9, color: "var(--muted)" }}>+{fragmentedWords.length - 5} more</span>}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-Components ────────────────────────────────────────────────

function HeadThumb({ matrix, active, boost, onClick }: { matrix: number[][]; active: boolean; boost: boolean; onClick: () => void }) {
  const dim = Math.min(matrix.length, 10);
  const step = Math.max(1, Math.floor(matrix.length / dim));
  const rows = Array.from({ length: dim }, (_, i) => matrix[Math.min(i * step, matrix.length - 1)].slice(0, dim));
  return (
    <div
      onClick={onClick}
      style={{
        borderRadius: 8, overflow: "hidden", cursor: "pointer",
        boxShadow: active ? "0 0 0 2px #8b5cf6" : "none",
        transform: active ? "scale(1.08)" : "scale(1)",
        transition: "transform 0.15s, box-shadow 0.15s",
      }}
    >
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex" }}>
          {row.map((v, ci) => (
            <div key={ci} style={{ width: 10, height: 6, backgroundColor: attnColor(v, boost) }} />
          ))}
        </div>
      ))}
    </div>
  );
}

function Heatmap({ tokens, matrix, boost }: { tokens: string[]; matrix: number[][]; boost: boolean }) {
  const sz = 26;
  const leftWidth = 110;
  return (
    <div style={{ overflowX: "auto", overflowY: "auto", maxHeight: "65vh", paddingRight: sz }}>
      <div style={{ display: "inline-flex", flexDirection: "column", gap: 1 }}>
        {/* column headers */}
        <div style={{ display: "flex", gap: 1, paddingLeft: leftWidth + 4, marginBottom: 2 }}>
          {tokens.map((t, i) => (
            <div key={i} style={{
              width: sz, fontSize: 8, color: "var(--muted)",
              fontFamily: "var(--font-mono, monospace)",
              writingMode: "vertical-rl", transform: "rotate(180deg)",
              maxHeight: 72, overflow: "hidden", textAlign: "left",
            }}>{t}</div>
          ))}
        </div>
        {matrix.map((row, ri) => (
          <div key={ri} style={{ display: "flex", gap: 1, alignItems: "center" }}>
            <div style={{ width: leftWidth, textAlign: "right", paddingRight: 4, fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)", overflow: "hidden", flexShrink: 0, textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {tokens[ri]}
            </div>
            {row.map((v, ci) => (
              <div key={ci} className="hm-cell" style={{ width: sz, height: sz, backgroundColor: attnColor(v, boost), flexShrink: 0 }}>
                <span className="tip">{v.toFixed(3)}</span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Bipartite token flow diagram for a selected token */
function TokenFlow({ tokens, row, fromIdx, onSelect }: { tokens: string[]; row: number[]; fromIdx: number; onSelect: (i: number) => void }) {
  const W = 520, H = Math.max(tokens.length * 28, 200);
  const cx = W / 2;
  const yFor = (i: number) => 14 + i * (H / tokens.length);

  return (
    <svg width={W} height={H} style={{ overflow: "visible", display: "block", margin: "0 auto" }}>
      {/* flow lines */}
      {tokens.map((_, ti) => {
        const v = row[ti];
        if (v < 0.01) return null;
        const x1 = 100, y1 = yFor(fromIdx);
        const x2 = W - 100, y2 = yFor(ti);
        return (
          <line key={ti} x1={x1} y1={y1} x2={x2} y2={y2}
            stroke={flowColor(v)} strokeWidth={1 + v * 6}
            strokeLinecap="round" opacity={0.85}
          />
        );
      })}
      {/* source tokens (left) */}
      {tokens.map((t, ti) => (
        <g key={`L${ti}`} onClick={() => onSelect(ti)} style={{ cursor: "pointer" }}>
          <rect x={4} y={yFor(ti) - 11} width={92} height={22} rx={5}
            fill={ti === fromIdx ? "rgba(59,130,246,0.25)" : "rgba(255,255,255,0.04)"}
            stroke={ti === fromIdx ? "#3b82f6" : "rgba(255,255,255,0.1)"} strokeWidth={1}
          />
          <text x={50} y={yFor(ti) + 4} textAnchor="middle" fontSize={9}
            fontFamily="var(--font-mono, monospace)" fill={ti === fromIdx ? "#93c5fd" : "#6b7280"}
          >{t.length > 10 ? t.slice(0, 9)+"…" : t}</text>
        </g>
      ))}
      {/* target tokens (right) */}
      {tokens.map((t, ti) => {
        const v = row[ti];
        return (
          <g key={`R${ti}`} onClick={() => onSelect(ti)} style={{ cursor: "pointer" }}>
            <rect x={W - 96} y={yFor(ti) - 11} width={92} height={22} rx={5}
              fill={`rgba(139,92,246,${0.05 + v * 0.35})`}
              stroke={`rgba(139,92,246,${0.15 + v * 0.6})`} strokeWidth={1}
            />
            <text x={W - 50} y={yFor(ti) + 4} textAnchor="middle" fontSize={9}
              fontFamily="var(--font-mono, monospace)" fill={v > 0.1 ? "#c4b5fd" : "#4b5563"}
            >{t.length > 10 ? t.slice(0, 9)+"…" : t}</text>
            {/* weight label */}
            {v > 0.05 && (
              <text x={W - 102} y={yFor(ti) + 4} textAnchor="end" fontSize={8}
                fontFamily="var(--font-mono, monospace)" fill="#7c3aed"
              >{v.toFixed(2)}</text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

/** Bar chart for practical limits */
function LimitsChart({ data }: { data: LimitRow[] }) {
  const maxLatency = Math.max(...data.filter(r => r.latency_ms !== null).map(r => r.latency_ms!), 1);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      {data.map(r => (
        <div key={r.seq_len} style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ width: 60, fontSize: 12, color: "var(--muted)", fontFamily: "var(--font-mono)", textAlign: "right", flexShrink: 0 }}>
            {r.seq_len}
          </div>
          <div style={{ flex: 1, background: "rgba(255,255,255,0.04)", borderRadius: 6, height: 24, position: "relative", overflow: "hidden" }}>
            {r.status === "ok" && r.latency_ms !== null ? (
              <div style={{
                height: "100%",
                width: `${(r.latency_ms / maxLatency) * 100}%`,
                background: "linear-gradient(90deg, #2563eb, #7c3aed)",
                borderRadius: 6,
                transition: "width 0.6s ease",
              }} />
            ) : (
              <div style={{ height: "100%", display: "flex", alignItems: "center", paddingLeft: 8 }}>
                <span style={{ fontSize: 10, color: "#ef4444", fontFamily: "var(--font-mono)" }}>OOM / Error</span>
              </div>
            )}
          </div>
          <div style={{ width: 80, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", textAlign: "right", flexShrink: 0 }}>
            {r.status === "ok" && r.latency_ms !== null ? `${r.latency_ms.toFixed(1)} ms` : "—"}
          </div>
          {r.memory_mb !== null && (
            <div style={{ width: 70, fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--muted)", textAlign: "right", flexShrink: 0 }}>
              {r.memory_mb.toFixed(0)} MB
            </div>
          )}
        </div>
      ))}
      <div style={{ display: "flex", gap: 12, paddingLeft: 72 }}>
        <span style={{ fontSize: 10, color: "var(--muted)" }}>← Sequence length (tokens)</span>
        <span style={{ fontSize: 10, color: "var(--muted)", marginLeft: "auto" }}>Latency (ms)</span>
      </div>
    </div>
  );
}

// ── Main Page ─────────────────────────────────────────────────────

export default function Home() {
  const [text, setText] = useState(
    "The self-attention mechanism allows every token to attend to every other token, creating rich bidirectional contextual representations."
  );
  const [modelId, setModelId] = useState(MODELS[0].id);
  const [tab, setTab] = useState<Tab>("attention");

  // Attention tab state
  const [attnLoading, setAttnLoading] = useState(false);
  const [attnError, setAttnError] = useState<string | null>(null);
  const [attnData, setAttnData] = useState<AttData | null>(null);
  const [selLayer, setSelLayer] = useState(0);
  const [selHead, setSelHead] = useState(-1);
  const [selToken, setSelToken] = useState(0);
  const [boostSignals, setBoostSignals] = useState(false);

  const [limitsLoading, setLimitsLoading] = useState(false);
  const [limitsError, setLimitsError] = useState<string | null>(null);
  const [limitsData, setLimitsData] = useState<LimitsData | null>(null);

  const selectedModel = MODELS.find(m => m.id === modelId) ?? MODELS[0];

  const fetchLimits = async (mId: string) => {
    const m = MODELS.find(x => x.id === mId) ?? MODELS[0];
    const seqLengths = SEQ_LENGTHS.filter(l => l <= m.maxLen);
    if (m.maxLen >= 1024 && !seqLengths.includes(1024)) seqLengths.push(1024);
    const res = await fetch(`${API}/api/limits`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: mId, seq_lengths: seqLengths }),
    });
    if (!res.ok) return null;
    return await res.json();
  };

  // ── Fetch attention & limits ──
  const analyzeAttention = useCallback(async () => {
    setAttnLoading(true);
    setAttnError(null);
    setLimitsError(null);
    try {
      const res1 = await fetch(`${API}/api/attention`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model_name: modelId }),
      });
      if (!res1.ok) { const b = await res1.json().catch(() => ({})); throw new Error(b.detail ?? `HTTP ${res1.status}`); }
      const d1: AttData = await res1.json();
      setAttnData(d1);

      const lim1 = await fetchLimits(modelId);
      if (lim1) setLimitsData(lim1);

      setSelLayer(0); setSelHead(0); setSelToken(0);
    } catch (e) {
      setAttnError(e instanceof Error ? e.message : String(e));
    } finally {
      setAttnLoading(false);
    }
  }, [text, modelId, tab]);

  // ── Stats for current head ──
  const stats = useMemo(() => {
    if (!attnData) return null;
    const layerArr = attnData.attention[selLayer];
    let matrix: number[][];

    if (selHead === -1) {
      // Average across all heads
      const numHeads = layerArr.length;
      const seqLen = layerArr[0].length;
      matrix = Array.from({ length: seqLen }, (_, i) => 
        Array.from({ length: seqLen }, (_, j) => 
          layerArr.reduce((sum, h) => sum + h[i][j], 0) / numHeads
        )
      );
    } else {
      matrix = layerArr[selHead];
    }

    const avgEnt = matrix.reduce((s, r) => s + entropy(r), 0) / matrix.length;
    const flat = matrix.flat();
    const maxVal = Math.max(...flat);
    let mi = 0, mj = 0;
    matrix.forEach((row, i) => row.forEach((v, j) => { if (v > matrix[mi][mj]) { mi = i; mj = j; } }));
    return { avgEnt, maxVal, topFrom: attnData.tokens[mi], topTo: attnData.tokens[mj], matrix };
  }, [attnData, selLayer, selHead]);

  const currentMatrix = stats?.matrix ?? [];
  const flowRow = currentMatrix[selToken] ?? [];

  // ── Tab button ──
  const TabBtn = ({ id, label, icon }: { id: Tab; label: string; icon: string }) => (
    <button
      onClick={() => setTab(id)}
      style={{
        padding: "8px 20px",
        borderRadius: 10,
        border: "1px solid",
        borderColor: tab === id ? "var(--accent-purple)" : "var(--border)",
        background: tab === id ? "rgba(139,92,246,0.18)" : "transparent",
        color: tab === id ? "#c4b5fd" : "var(--muted)",
        fontWeight: tab === id ? 600 : 400,
        fontSize: 13, cursor: "pointer",
        transition: "all 0.15s",
        display: "flex", alignItems: "center", gap: 6,
      }}
    >
      <span>{icon}</span>{label}
    </button>
  );

  return (
    <main style={{ minHeight: "100vh", padding: "40px 20px", display: "flex", flexDirection: "column", alignItems: "center" }}>
      <div style={{ width: "100%", maxWidth: 1160, display: "flex", flexDirection: "column", gap: 32 }}>

        {/* ── Header ── */}
        <header style={{ textAlign: "center" }}>
          <div style={{ fontSize: 11, letterSpacing: "0.15em", textTransform: "uppercase", color: "var(--accent-purple)", fontWeight: 600, marginBottom: 8 }}>
            Research Tool
          </div>
          <h1 className="gradient-text" style={{ fontSize: "clamp(1.8rem, 4vw, 3rem)", fontWeight: 800, lineHeight: 1.15, marginBottom: 10 }}>
            Encoder Attention Explorer
          </h1>
          <p style={{ color: "var(--muted)", fontSize: 15, maxWidth: 580, margin: "0 auto" }}>
            Probe bidirectional self-attention in BERT, RoBERTa, DeBERTa&nbsp;&amp;&nbsp;ModernBERT.
            Explore attention matrices, practical context limits, and token-level information flow.
          </p>
        </header>

        {/* ── Input Panel ── */}
        <section className="glass" style={{ padding: 28 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 240px", gap: 20 }}>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em" }}>CORPUS TEXT</label>
              <textarea value={text} onChange={e => setText(e.target.value)} rows={3}
                style={{ width: "100%", padding: "12px 14px", fontSize: 14, resize: "vertical" }}
                placeholder="Enter any text to visualize attention…" />
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                <label style={{ fontSize: 11, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em" }}>MODEL</label>
                <select value={modelId} onChange={e => setModelId(e.target.value)} style={{ padding: "10px 12px", fontSize: 13 }}>
                  {MODELS.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
                </select>
                <div style={{ fontSize: 10, color: "var(--muted)", fontFamily: "var(--font-mono, monospace)" }}>{selectedModel.desc}</div>
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button className="glow-btn" onClick={analyzeAttention} disabled={attnLoading || !text.trim()}
                  style={{ flex: 1, padding: "11px 0", fontSize: 13, display: "flex", alignItems: "center", justifyContent: "center", gap: 7 }}>
                  {attnLoading && <span className="spinner" />}
                  {attnLoading ? "Analyzing…" : "Analyze"}
                </button>
              </div>
            </div>
          </div>
          {(attnError || limitsError) && (
            <div style={{ marginTop: 14, padding: "10px 14px", borderRadius: 8, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#fca5a5", fontSize: 12 }}>
              ⚠ {attnError ?? limitsError}
            </div>
          )}
        </section>

        {/* ── Corpus Verdict (Decision Engine) ── */}
        {attnData && (
          <section className="glass fade-up" style={{ padding: "18px 24px", display: "flex", alignItems: "center", gap: 18, background: "rgba(139, 92, 246, 0.06)", border: "1px solid rgba(139, 92, 246, 0.25)" }}>
            <div style={{ fontSize: 32 }}>{getFragmentation(attnData.tokens).fragRatio > 1.35 ? "⚠️" : "🚀"}</div>
            <div>
              <div style={{ fontSize: 12, fontWeight: 800, color: "#c4b5fd", letterSpacing: "0.08em", marginBottom: 4 }}>CORPUS COMPATIBILITY VERDICT</div>
              <p style={{ fontSize: 13, color: "var(--muted)", margin: 0, lineHeight: 1.5 }}>
                {getFragmentation(attnData.tokens).fragRatio > 1.35 
                  ? <span>Your text contains complex or domain-specific language that highly fragments <strong>{selectedModel.label}</strong>'s vocabulary ({getFragmentation(attnData.tokens).fragRatio.toFixed(2)} tokens/word). You will lose early-layer semantic reasoning capacity. Consider swapping to <strong>ModernBERT</strong> or <strong>BGE-M3</strong>.</span>
                  : <span><strong>{selectedModel.label}</strong> natively understands your text's vocabulary perfectly ({getFragmentation(attnData.tokens).fragRatio.toFixed(2)} tokens/word). Deep semantic routing will be highly efficient. This is an excellent architectural choice for this corpus.</span>}
              </p>
            </div>
          </section>
        )}

        {/* ── Tabs ── */}
        {(attnData || limitsData) && (
          <div style={{ display: "flex", gap: 10 }}>
            {attnData && <TabBtn id="attention" label="Attention Matrix" icon="🧠" />}
            {attnData && <TabBtn id="tokenization" label="Vocabulary Fit" icon="🔬" />}
            {attnData && <TabBtn id="flow" label="Token Flow" icon="🌊" />}
            {limitsData && <TabBtn id="limits" label="Practical Limits" icon="⚡" />}
            <TabBtn id="leaderboard" label="Leaderboards" icon="🏆" />
            {limitsData && <TabBtn id="limits" label="Practical Limits" icon="⚡" />}
          </div>
        )}

        {/* ══ TOKENIZATION ANALYSIS TAB ══ */}
        {tab === "tokenization" && attnData && (
          <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 20 }}>
            <section className="glass" style={{ padding: 22 }}>
              <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 12 }}>Fragmentation Map: {selectedModel.label}</h2>
              {attnData.tokens.some(t => t.includes("Ġ")) && (
                <div style={{ marginBottom: 14, padding: "10px 14px", background: "rgba(59,130,246,0.1)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: 8, fontSize: 12, color: "#93c5fd", lineHeight: 1.5 }}>
                  <strong style={{ color: "#bfdbfe" }}>Tokenizer Note:</strong> The <code>Ġ</code> character is this architecture's internal BPE marker for a preceding space. It mathematically bakes spacing directly into the vocabulary to prevent boundary bugs.
                </div>
              )}
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 4px", background: "rgba(0,0,0,0.15)", padding: 16, borderRadius: 12, lineHeight: 2 }}>
                {(() => {
                  const isFragToken = new Array(attnData.tokens.length).fill(false);
                  let currentWordStart = -1;
                  for (let i = 0; i < attnData.tokens.length; i++) {
                    const t = attnData.tokens[i];
                    const isSpecial = ["[CLS]","[SEP]","<s>","</s>","<pad>","[MASK]"].includes(t);
                    if (isSpecial) continue;
                    
                    const isSub = t.startsWith("##") || (!t.startsWith("Ġ") && !t.startsWith(" ") && i > 0 && !["[CLS]","[SEP]","<s>","</s>","<pad>","[MASK]"].includes(attnData.tokens[i]));
                    if (!isSub) {
                      currentWordStart = i;
                    } else {
                      isFragToken[i] = true;
                      if (currentWordStart !== -1) isFragToken[currentWordStart] = true;
                    }
                  }

                  return attnData.tokens.map((t, i) => {
                    const isSpecial = ["[CLS]","[SEP]","<s>","</s>","<pad>","[MASK]"].includes(t);
                    const isBroken = isFragToken[i];
                    return (
                      <span key={i} style={{
                        padding: "2px 6px",
                        borderRadius: 4,
                        fontSize: 13,
                        fontFamily: "var(--font-mono, monospace)",
                        background: isSpecial ? "rgba(255,255,255,0.05)" : isBroken ? "rgba(248,113,113,0.15)" : "rgba(52,211,153,0.1)",
                        color: isSpecial ? "var(--muted)" : isBroken ? "#fca5a5" : "#6ee7b7",
                        border: "1px solid",
                        borderColor: isSpecial ? "transparent" : isBroken ? "rgba(248,113,113,0.3)" : "rgba(52,211,153,0.2)",
                      }}>
                        {t}
                      </span>
                    );
                  });
                })()}
              </div>
            </section>

            <section className="glass" style={{ padding: 22 }}>
              <TokenHealth tokens={attnData.tokens} />
              <div style={{ height: "1px", background: "var(--border)", opacity: 0.4, margin: "20px 0" }} />
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent-purple)", marginBottom: 10 }}>SCIENTIST'S NOTE</div>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.6 }}>
                Every split token represents an "Attention Penalty." When a model has high overhead (red markers), it must spend its first 2-3 layers performing sub-word assembly rather than higher-order semantic reasoning. If your domain text shows &gt;40% overhead, consider a model with a larger vocabulary (like <strong>ModernBERT</strong> or <strong>BGE-M3</strong>).
              </p>
            </section>
          </div>
        )}

        {/* ══ LEADERBOARDS TAB ══ */}
        {tab === "leaderboard" && (
          <div className="fade-up" style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <section className="glass" style={{ padding: 32 }}>
              <div style={{ marginBottom: 24 }}>
                <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
                  NLP Performance Leaderboards
                </h2>
                <p style={{ fontSize: 12, color: "var(--muted)" }}>
                  Comparing the objective reasoning capabilities (GLUE) and semantic retrieval capabilities (MTEB) of historical and modern encoder architectures.
                </p>
              </div>

              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)", background: "rgba(255,255,255,0.02)" }}>
                      <th style={{ textAlign: "left", padding: "12px", color: "var(--muted)", fontWeight: 600 }}>Architecture</th>
                      <th style={{ textAlign: "left", padding: "12px", color: "var(--muted)", fontWeight: 600 }}>Params</th>
                      <th style={{ textAlign: "left", padding: "12px", color: "var(--muted)", fontWeight: 600 }}>Era</th>
                      <th style={{ textAlign: "left", padding: "12px", color: "var(--muted)", fontWeight: 600 }}>GLUE (Avg)</th>
                      <th style={{ textAlign: "left", padding: "12px", color: "var(--muted)", fontWeight: 600 }}>MTEB (Retrieval)</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#cbd5e1" }}>DeBERTa v3 Base</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>184M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2021</td>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#34d399" }}>91.4</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>—</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", background: "rgba(139,92,246,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#a78bfa" }}>ModernBERT Base</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>149M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2024</td>
                      <td style={{ padding: "12px", fontWeight: 700 }}>87.2</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>—</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#cbd5e1" }}>RoBERTa Base</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>125M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2019</td>
                      <td style={{ padding: "12px" }}>86.8</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>—</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", background: "rgba(59,130,246,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#93c5fd" }}>GTE-ModernBERT Base</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>149M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2025</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>—</td>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#34d399" }}>60.5</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "#cbd5e1" }}>BGE-M3</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>568M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2024</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>—</td>
                      <td style={{ padding: "12px", fontWeight: 700 }}>58.2</td>
                    </tr>
                    <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "12px", fontWeight: 700, color: "var(--muted)" }}>BERT Base</td>
                      <td style={{ padding: "12px", color: "var(--muted)", fontFamily: "var(--font-mono)" }}>110M</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>2018</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>79.6</td>
                      <td style={{ padding: "12px", color: "var(--muted)" }}>47.8</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}
        {tab === "attention" && attnData && (
          <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "start" }}>

            {/* Left: controls */}
            <section className="glass" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 24 }}>

              {/* Stats chips */}
              {stats && (
                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em" }}>STATS</div>
                  <div className="chip" style={{ fontSize: 11 }}>H: <strong>{stats.avgEnt.toFixed(2)} nats</strong></div>
                  <div className="chip" style={{ fontSize: 11 }}>Peak: <strong>{stats.topFrom}</strong> → <strong>{stats.topTo}</strong></div>
                  <div className="chip" style={{ fontSize: 11 }}>Max: <strong>{stats.maxVal.toFixed(3)}</strong></div>
                </div>
              )}

              {/* Layer selector */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 8 }}>LAYER</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {attnData.attention.map((_, i) => (
                    <button key={i} onClick={() => setSelLayer(i)} style={{
                      width: 28, height: 28, borderRadius: 7, border: "none", cursor: "pointer",
                      fontSize: 11, fontWeight: 600,
                      background: selLayer === i ? "linear-gradient(135deg,#2563eb,#7c3aed)" : "rgba(255,255,255,0.06)",
                      color: selLayer === i ? "#fff" : "var(--muted)",
                      transition: "all 0.12s",
                    }}>{i + 1}</button>
                  ))}
                </div>
              </div>

              {/* Head thumbnails */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 8 }}>HEAD</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  <button 
                    onClick={() => setSelHead(-1)}
                    style={{
                      gridColumn: "span 2",
                      padding: "8px",
                      borderRadius: 8,
                      border: "1px solid",
                      borderColor: selHead === -1 ? "#8b5cf6" : "var(--border)",
                      background: selHead === -1 ? "rgba(139,92,246,0.15)" : "rgba(255,255,255,0.03)",
                      color: selHead === -1 ? "#c4b5fd" : "var(--muted)",
                      fontSize: 11, fontWeight: 700, cursor: "pointer",
                      transition: "all 0.15s",
                    }}
                  >
                    Layer Average
                  </button>
                  {attnData.attention[selLayer].map((hm, hi) => (
                    <div key={hi} style={{ display: "flex", flexDirection: "column", gap: 3, alignItems: "center" }}>
                      <HeadThumb matrix={hm} active={selHead === hi} boost={boostSignals} onClick={() => setSelHead(hi)} />
                      <span style={{ fontSize: 9, color: selHead === hi ? "#a78bfa" : "var(--muted)" }}>H{hi + 1}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Legend */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 6 }}>WEIGHT</div>
                <div className="legend-bar" />
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: 9, color: "var(--muted)", fontFamily: "var(--font-mono)", marginTop: 3 }}>
                  <span>0.000</span><span>1.000</span>
                </div>
              </div>
            </section>

            {/* Right: heatmap */}
            <section className="glass" style={{ padding: 22 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 16 }}>
                <div>
                  <h2 style={{ fontSize: 16, fontWeight: 700 }}>
                    Layer {selLayer + 1} · {selHead === -1 ? "Average" : `Head ${selHead + 1}`}
                  </h2>
                  <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                    <p style={{ fontSize: 11, color: "var(--muted)" }}>{selectedModel.label} — hover cells for values</p>
                    {/* Architectural Insights */}
                    {modelId.includes("ModernBERT") && (
                      <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, background: (selLayer + 1) % 3 === 0 ? "rgba(139,92,246,0.2)" : "rgba(59,130,246,0.2)", color: (selLayer + 1) % 3 === 0 ? "#c4b5fd" : "#93c5fd", fontWeight: 700, border: "1px solid currentColor" }}>
                        {(selLayer + 1) % 3 === 0 ? "GLOBAL LAYER" : "LOCAL WINDOW LAYER"}
                      </span>
                    )}
                    {modelId.includes("bigbird") && (
                      <span style={{ fontSize: 9, padding: "1px 6px", borderRadius: 4, background: "rgba(16,185,129,0.2)", color: "#6ee7b7", fontWeight: 700, border: "1px solid currentColor" }}>
                        SPARSE BLOCK ATTENTION
                      </span>
                    )}
                  </div>
                </div>
                <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
                  <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "var(--font-mono)", textAlign: "right" }}>
                    {attnData.tokens.length} tokens
                  </div>
                  <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, cursor: "pointer", color: boostSignals ? "#c4b5fd" : "var(--muted)" }}>
                    <input type="checkbox" checked={boostSignals} onChange={e => setBoostSignals(e.target.checked)} />
                    Boost Faint Signals
                  </label>
                </div>
              </div>
              <Heatmap tokens={attnData.tokens} matrix={currentMatrix} boost={boostSignals} />

              <div style={{ marginTop: 32, borderTop: "1px solid var(--border)", paddingTop: 24 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent-purple)", marginBottom: 12 }}>HOW TO INTERPRET PATTERNS</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>VERTICAL LINES (SINKS)</div>
                    <p style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>
                      Probability mass concentrated on <strong>[CLS]</strong> or punctuation. These tokens act as "pressure release valves" when no semantic relationship is found.
                    </p>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>DIAGONAL LINES (LOCALITY)</div>
                    <p style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>
                      Strong attention to immediate neighbors. High in early layers, this indicates the model is parsing <strong>local syntax</strong> and grammar.
                    </p>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>SPARSE DOTS (GLOBAL)</div>
                    <p style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>
                      Strong links between distant words. Often represents <strong>coreference resolution</strong> (e.g., a pronoun attending to its noun) or entity relationships.
                    </p>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, fontWeight: 700, color: "var(--text)", marginBottom: 4 }}>BLOCK PATTERNS</div>
                    <p style={{ fontSize: 11, color: "var(--muted)", lineHeight: 1.5 }}>
                      A cluster of tokens attending to each other. Typical for <strong>multi-word entities</strong> or phrases that the model treats as a single semantic unit.
                    </p>
                  </div>
                </div>
              </div>
            </section>
          </div>
        )}

        {/* ══ TOKEN FLOW TAB ══ */}
        {tab === "flow" && attnData && (
          <div className="fade-up" style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20, alignItems: "start" }}>

            <section className="glass" style={{ padding: 18, display: "flex", flexDirection: "column", gap: 18 }}>
              {/* Layer/head controls reused */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 8 }}>LAYER</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {attnData.attention.map((_, i) => (
                    <button key={i} onClick={() => setSelLayer(i)} style={{
                      width: 28, height: 28, borderRadius: 7, border: "none", cursor: "pointer", fontSize: 11, fontWeight: 600,
                      background: selLayer === i ? "linear-gradient(135deg,#2563eb,#7c3aed)" : "rgba(255,255,255,0.06)",
                      color: selLayer === i ? "#fff" : "var(--muted)", transition: "all 0.12s",
                    }}>{i + 1}</button>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 8 }}>HEAD</div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                  {attnData.attention[selLayer].map((_, hi) => (
                    <button key={hi} onClick={() => setSelHead(hi)} style={{
                      width: 28, height: 28, borderRadius: 7, border: "none", cursor: "pointer", fontSize: 11, fontWeight: 600,
                      background: selHead === hi ? "linear-gradient(135deg,#2563eb,#7c3aed)" : "rgba(255,255,255,0.06)",
                      color: selHead === hi ? "#fff" : "var(--muted)", transition: "all 0.12s",
                    }}>{hi + 1}</button>
                  ))}
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.08em", marginBottom: 8 }}>SOURCE TOKEN</div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4, maxHeight: 260, overflowY: "auto" }}>
                  {attnData.tokens.map((t, ti) => (
                    <button key={ti} onClick={() => setSelToken(ti)} style={{
                      textAlign: "left", padding: "5px 10px", borderRadius: 6, border: "1px solid",
                      borderColor: selToken === ti ? "var(--accent-blue)" : "var(--border)",
                      background: selToken === ti ? "rgba(59,130,246,0.15)" : "transparent",
                      color: selToken === ti ? "#93c5fd" : "var(--muted)",
                      fontSize: 11, fontFamily: "var(--font-mono)", cursor: "pointer", transition: "all 0.12s",
                    }}>{t}</button>
                  ))}
                </div>
              </div>
            </section>

            <section className="glass" style={{ padding: 22 }}>
              <h2 style={{ fontSize: 16, fontWeight: 700, marginBottom: 4 }}>
                Token Flow · <span style={{ color: "#93c5fd" }}>{attnData.tokens[selToken]}</span>
              </h2>
              <p style={{ fontSize: 11, color: "var(--muted)", marginBottom: 20 }}>
                Line thickness and opacity encode attention weight from the selected source token. Click any token to pivot.
              </p>
              <TokenFlow tokens={attnData.tokens} row={flowRow} fromIdx={selToken} onSelect={setSelToken} />
            </section>
          </div>
        )}

        {/* ══ PRACTICAL LIMITS TAB ══ */}
        {tab === "limits" && limitsData && (
          <div className="fade-up">
            <section className="glass" style={{ padding: 28 }}>
              <div style={{ marginBottom: 24 }}>
                <h2 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>
                  Practical Limits — {MODELS.find(m => m.id === limitsData.model)?.label ?? limitsData.model}
                </h2>
                <p style={{ fontSize: 12, color: "var(--muted)" }}>
                  Forward-pass latency (ms) and peak GPU memory (MB) across sequence lengths. OOM = out-of-memory.
                </p>
              </div>
              <LimitsChart data={limitsData.results} />

              {/* Raw table */}
              <div style={{ marginTop: 28, overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, fontFamily: "var(--font-mono, monospace)" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid var(--border)" }}>
                      {["Seq Len", "Latency (ms)", "Memory (MB)", "Status"].map(h => (
                        <th key={h} style={{ textAlign: "left", padding: "8px 12px", color: "var(--muted)", fontWeight: 600 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {limitsData.results.map(r => (
                      <tr key={r.seq_len} style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                        <td style={{ padding: "8px 12px" }}>{r.seq_len}</td>
                        <td style={{ padding: "8px 12px" }}>{r.latency_ms !== null ? r.latency_ms.toFixed(2) : "—"}</td>
                        <td style={{ padding: "8px 12px" }}>{r.memory_mb !== null ? r.memory_mb.toFixed(1) : "—"}</td>
                        <td style={{ padding: "8px 12px", color: r.status === "ok" ? "#34d399" : "#f87171" }}>{r.status}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* ── Selection Guide ── */}
        <section className="glass" style={{ padding: 28 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 18 }}>
            <span style={{ fontSize: 20 }}>📘</span>
            <h2 style={{ fontSize: 18, fontWeight: 700 }}>Practical Model Selection Guide</h2>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 24 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#3b82f6", marginBottom: 6 }}>NLU & CLASSIFICATION</div>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
                Use <strong>DeBERTa v3</strong>. Its disentangled attention mechanism is specifically engineered to understand relative positioning, making it the top performer for GLUE benchmarks.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#8b5cf6", marginBottom: 6 }}>LONG CONTEXTS (8k+)</div>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
                Use <strong>ModernBERT</strong>. Its alternating local-global attention schedule and FlashAttention-3 integration allow it to handle 8k context with higher speed than legacy encoders.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#10b981", marginBottom: 6 }}>RETRIEVAL & EMBEDDINGS</div>
              <p style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5 }}>
                Use <strong>BGE-M3</strong>. It is pre-trained with a focus on dense retrieval and handles multi-lingual inputs with extremely low fragmentation.
              </p>
            </div>
          </div>
        </section>

        {/* ── Empty state ── */}
        {!attnData && !limitsData && (
          <div style={{ textAlign: "center", padding: "60px 0", color: "var(--muted)" }}>
            <div style={{ fontSize: 48, marginBottom: 12 }}>🔬</div>
            <p style={{ fontSize: 14 }}>
              Click <strong style={{ color: "var(--text)" }}>Analyze</strong> to explore attention matrices
              and benchmark sequence length limits.
            </p>
          </div>
        )}

        <footer style={{ textAlign: "center", fontSize: 11, color: "var(--muted)", paddingTop: 20, borderTop: "1px solid var(--border)" }}>
          Encoder Attention Explorer · Research tool for the arXiv paper on encoder-only transformer attention.
        </footer>
      </div>
    </main>
  );
}
