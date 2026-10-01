import React, { useState, useEffect } from 'react';
import {
  ShieldAlert, ShieldCheck, Activity, Network,
  Target, Loader, Zap, MessageSquare
} from 'lucide-react';
import { fetchExplainability, fetchTracePattern, fetchLLMExplainability } from '../utils/apiService';

const PATTERN_META = {
  cycle:           { label: 'Cycle',           color: '#fb923c', desc: 'Circular fund flow' },
  fan_out:         { label: 'Fan-Out',         color: '#e85454', desc: 'Dispersed to many' },
  fan_in:          { label: 'Fan-In',          color: '#9b73f5', desc: 'Consolidated from many' },
  peeling_chain:   { label: 'Peeling Chain',   color: '#f0a142', desc: 'Sequential layering' },
  direct_transfer: { label: 'Direct Transfer', color: '#3dd68c', desc: 'Point-to-point' },
};

export default function Sidebar({ selectedNode, onTrace, jobId }) {
  const [activeTab, setActiveTab] = useState('SAGE');
  const [shapData, setShapData] = useState(null);
  const [isLoadingShap, setIsLoadingShap] = useState(false);
  const [tracePattern, setTracePattern] = useState(null);
  const [isTracing, setIsTracing] = useState(false);
  const [llmData, setLlmData] = useState(null);
  const [isLoadingLLM, setIsLoadingLLM] = useState(false);

  useEffect(() => {
    if (!selectedNode) {
      setShapData(null); setTracePattern(null); setLlmData(null);
      return;
    }
    setActiveTab('SAGE');
    setIsLoadingShap(true);
    fetchExplainability(jobId, selectedNode.id)
      .then(res => setShapData(res.top_risk_factors || []))
      .catch(() => setShapData([]))
      .finally(() => setIsLoadingShap(false));

    fetchTracePattern(jobId, selectedNode.id)
      .then(res => setTracePattern(res))
      .catch(() => setTracePattern(null));

    setLlmData(null);
  }, [selectedNode, jobId]);

  if (!selectedNode) {
    return (
      <div className="entity-panel">
        <div className="panel-header">
          <Target size={16} color="var(--text-muted)" />
          <h2>Entity Inspector</h2>
        </div>
        <div className="panel-body">
          <div style={{
            textAlign: 'center', padding: '40px 20px', color: 'var(--text-muted)',
            fontSize: '0.83rem', lineHeight: 1.7
          }}>
            <Network size={36} style={{ opacity: 0.2, marginBottom: 14 }} />
            <p>Click any wallet node in the graph to inspect its risk profile, GNN features, SHAP contributions, and AI explanation.</p>
          </div>
        </div>
      </div>
    );
  }

  const isAnomaly = selectedNode.is_anomaly === true;
  const riskPct = ((selectedNode.risk_probability || 0) * 100).toFixed(1);
  const patMeta = tracePattern ? (PATTERN_META[tracePattern.pattern] || { label: tracePattern.pattern, color: '#8892a4', desc: '' }) : null;

  const handleTrace = async () => {
    setIsTracing(true);
    try { await onTrace(selectedNode.id); } finally { setIsTracing(false); }
  };

  const handleLLM = async () => {
    setIsLoadingLLM(true);
    try {
      const res = await fetchLLMExplainability(jobId, selectedNode.id);
      setLlmData(res);
    } catch {
      setLlmData({ error: true, message: 'Failed to generate. Ensure backend model is loaded.' });
    } finally {
      setIsLoadingLLM(false);
    }
  };

  return (
    <div className="entity-panel">
      <div className="panel-header">
        {isAnomaly
          ? <ShieldAlert size={16} color="var(--danger)" />
          : <ShieldCheck size={16} color="var(--success)" />}
        <h2>Entity Inspector</h2>
      </div>

      <div className="panel-body">
        {/* ── Identity Card ── */}
        <div className="entity-header-card" style={{
          borderLeft: `3px solid ${isAnomaly ? 'var(--danger)' : 'var(--success)'}`
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <span className="entity-id">{selectedNode.id}</span>
            <span className={`entity-badge ${isAnomaly ? 'high' : 'normal'}`}>
              {isAnomaly ? `High Risk ${riskPct}%` : `Normal ${riskPct}%`}
            </span>
          </div>

          {patMeta && (
            <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
              <span className="pattern-tag" style={{
                background: patMeta.color + '18',
                color: patMeta.color,
                border: `1px solid ${patMeta.color}44`
              }}>
                {patMeta.label}
              </span>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                {patMeta.desc}
              </span>
              <span style={{ marginLeft: 'auto', fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Hops: <strong style={{ color: 'var(--text-secondary)' }}>{tracePattern.hop_count}</strong>
              </span>
            </div>
          )}
        </div>

        {/* ── Phase Tabs ── */}
        <div className="phase-tabs">
          {[
            { key: 'SAGE',      label: 'Phase 1', sub: 'GNN' },
            { key: 'ISOLATION', label: 'Phase 2', sub: 'Risk' },
            { key: 'EXPLAIN',   label: 'Phase 3', sub: 'SHAP' },
            { key: 'LLM',       label: 'Phase 4', sub: 'LLM' },
          ].map(t => (
            <button
              key={t.key}
              className={`phase-tab ${activeTab === t.key ? 'active' : ''}`}
              onClick={() => setActiveTab(t.key)}
            >
              {t.label}<br />{t.sub}
            </button>
          ))}
        </div>

        {/* ── Phase 1: GNN ── */}
        {activeTab === 'SAGE' && (
          <div className="data-card">
            <h3><Network size={13} /> GraphSAGE Features</h3>
            {[
              ['Total Connections', selectedNode.degree || (selectedNode.in_degree + selectedNode.out_degree) || 0],
              ['In / Out Degree', `${selectedNode.in_degree || 0} in / ${selectedNode.out_degree || 0} out`],
              ['PageRank', (selectedNode.pagerank || 0).toFixed(6)],
              ['Clustering Coeff.', (selectedNode.clustering_coefficient || 0).toFixed(4)],
              ['Betweenness', (selectedNode.betweenness_centrality || 0).toFixed(6)],
              ['Community ID', selectedNode.community_id ?? '—'],
            ].map(([label, val]) => (
              <div className="metric-row" key={label}>
                <span className="metric-label">{label}</span>
                <span className="metric-value">{val}</span>
              </div>
            ))}
          </div>
        )}

        {/* ── Phase 2: Risk ── */}
        {activeTab === 'ISOLATION' && (
          <div className="data-card" style={{ borderLeft: `3px solid ${isAnomaly ? 'var(--danger)' : 'var(--success)'}` }}>
            <h3><Activity size={13} /> ML Risk Scoring</h3>
            <div className="metric-row">
              <span className="metric-label">Risk Probability</span>
              <span className="metric-value" style={{
                fontSize: '1.15rem', fontWeight: 700,
                color: isAnomaly ? 'var(--danger)' : 'var(--success)'
              }}>{riskPct}%</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Classification</span>
              <span className="metric-value" style={{ color: isAnomaly ? 'var(--danger)' : 'var(--success)', fontWeight: 600 }}>
                {isAnomaly ? 'FLAGGED' : 'NORMAL'}
              </span>
            </div>
            {[
              ['Total Sent', `${(selectedNode.total_sent_sats || 0).toLocaleString()} sats`],
              ['Total Received', `${(selectedNode.total_received_sats || 0).toLocaleString()} sats`],
              ['Tx Count', selectedNode.tx_count || selectedNode.degree || 0],
            ].map(([label, val]) => (
              <div className="metric-row" key={label}>
                <span className="metric-label">{label}</span>
                <span className="metric-value">{val}</span>
              </div>
            ))}
          </div>
        )}

        {/* ── Phase 3: SHAP ── */}
        {activeTab === 'EXPLAIN' && (
          <div className="data-card" style={{ borderLeft: '3px solid var(--warning)' }}>
            <h3><Target size={13} /> SHAP Explainability</h3>
            {isLoadingShap ? (
              <div style={{ display: 'flex', justifyContent: 'center', padding: 20 }}>
                <Loader size={22} className="spinner" color="var(--warning)" />
              </div>
            ) : shapData && shapData.length > 0 ? (
              shapData.slice(0, 5).map((f, i) => {
                const isInc = f.direction === 'increases_risk';
                const absVal = Math.abs(f.shap_value || 0);
                const maxVal = 3;
                const barPct = Math.min(100, (absVal / maxVal) * 100);
                return (
                  <div className="shap-bar-row" key={i}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span className="shap-name">{f.feature || f.feature_name}</span>
                      <span style={{
                        fontSize: '0.68rem', fontWeight: 600,
                        color: isInc ? 'var(--danger)' : 'var(--success)'
                      }}>
                        {isInc ? '▲ Risk' : '▼ Risk'}
                      </span>
                    </div>
                    <div className="shap-bar-wrap">
                      <div className="shap-bar-track">
                        <div className="shap-bar-fill" style={{
                          width: `${barPct}%`,
                          background: isInc ? 'var(--danger)' : 'var(--success)'
                        }} />
                      </div>
                      <span className="shap-val">{(f.shap_value || 0).toFixed(3)}</span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                      Value: <strong style={{ color: 'var(--text-secondary)' }}>
                        {Number(f.raw_value || f.feature_value || 0).toLocaleString()}
                      </strong>
                    </div>
                  </div>
                );
              })
            ) : (
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>No SHAP data available.</p>
            )}
          </div>
        )}

        {/* ── Phase 4: LLM ── */}
        {activeTab === 'LLM' && (
          <div className="data-card" style={{ borderLeft: '3px solid var(--purple)' }}>
            <h3><MessageSquare size={13} /> LLM Analyst</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 12, lineHeight: 1.6 }}>
              On-device forensic summary. No data leaves your machine.
            </p>

            {!llmData && !isLoadingLLM && (
              <button className="action-btn primary" onClick={handleLLM}
                style={{ background: 'var(--purple)' }}>
                <Zap size={14} /> Generate Explanation
              </button>
            )}

            {isLoadingLLM && (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, padding: 16 }}>
                <Loader size={24} className="spinner" color="var(--purple)" />
                <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  Generating… up to 30s
                </span>
              </div>
            )}

            {llmData && !llmData.error && (
              <div className="llm-box">
                <div className="llm-model-tag">
                  <Zap size={10} /> {llmData.model_used}
                  {llmData.cached && <span style={{ marginLeft: 6, opacity: 0.6 }}>· cached</span>}
                </div>
                <div style={{ whiteSpace: 'pre-wrap', color: 'var(--text-primary)', lineHeight: 1.7 }}>
                  {llmData.explanation}
                </div>
              </div>
            )}

            {llmData && llmData.error && (
              <div className="llm-error">{llmData.message}</div>
            )}
          </div>
        )}

        {/* ── Trace Button ── */}
        <button className="action-btn warning" onClick={handleTrace} disabled={isTracing}>
          <Zap size={14} />
          {isTracing ? 'Tracing…' : 'Dynamic Trace Flow'}
        </button>
      </div>
    </div>
  );
}
