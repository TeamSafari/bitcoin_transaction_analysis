import React, { useState, useEffect } from 'react';
import { ShieldAlert, ShieldCheck, Activity, Network, Target, ChevronRight, Loader, Zap, ArrowUpRight, ArrowDownLeft, MessageSquare } from 'lucide-react';
import { fetchExplainability, fetchTracePattern, fetchLLMExplainability } from '../utils/apiService';

export default function Sidebar({ selectedNode, onTrace, jobId }) {
  const [activeTab, setActiveTab] = useState('SAGE');
  const [shapData, setShapData] = useState(null);
  const [isLoadingShap, setIsLoadingShap] = useState(false);
  const [tracePattern, setTracePattern] = useState(null);
  const [isTracing, setIsTracing] = useState(false);
  const [llmExplainData, setLlmExplainData] = useState(null);
  const [isLoadingLLM, setIsLoadingLLM] = useState(false);

  useEffect(() => {
    if (!selectedNode) {
      setShapData(null);
      setTracePattern(null);
      setLlmExplainData(null);
      return;
    }

    // Reset and fetch fresh SHAP explainability for the selected node
    setIsLoadingShap(true);
    fetchExplainability(jobId, selectedNode.id)
      .then(res => {
        setShapData(res.top_risk_factors || []);
      })
      .catch(err => {
        console.warn('Explainability fetch error:', err);
        setShapData([]);
      })
      .finally(() => setIsLoadingShap(false));

    // Also fetch trace summary info
    fetchTracePattern(jobId, selectedNode.id)
      .then(res => {
        setTracePattern(res);
      })
      .catch(() => setTracePattern(null));
      
    // Clear LLM explanation on node switch
    setLlmExplainData(null);

  }, [selectedNode, jobId]);

  if (!selectedNode) {
    return (
      <div className="sidebar">
        <h2><Target color="#3b82f6" /> XAI Analyst View</h2>
        <p style={{ color: 'var(--text-muted)' }}>
          Click any wallet node or use the search bar to view its multi-stage GraphSAGE, ML risk scoring, and SHAP explainability.
        </p>
      </div>
    );
  }

  const isAnomaly = selectedNode.is_anomaly === true;
  const riskPct = ((selectedNode.risk_probability || 0) * 100).toFixed(1);

  const handleTraceClick = async () => {
    setIsTracing(true);
    try {
      await onTrace(selectedNode.id);
    } finally {
      setIsTracing(false);
    }
  };

  const handleFetchLLM = async () => {
    if (!selectedNode) return;
    setIsLoadingLLM(true);
    try {
      const res = await fetchLLMExplainability(jobId, selectedNode.id);
      setLlmExplainData(res);
    } catch (err) {
      console.error("LLM Explainability Error:", err);
      setLlmExplainData({ error: true, message: 'Failed to generate explanation. Ensure backend model is loaded.' });
    } finally {
      setIsLoadingLLM(false);
    }
  };

  return (
    <div className="sidebar">
      <h2>{isAnomaly ? <ShieldAlert color="#ef4444" /> : <ShieldCheck color="#3b82f6" />} Entity Profile</h2>
      
      <div className="panel-card" style={{ borderLeft: isAnomaly ? '4px solid #ef4444' : '4px solid #3b82f6' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <h3 style={{ margin: 0 }}>{selectedNode.id}</h3>
          <span className={`badge ${isAnomaly ? 'danger' : 'safe'}`}>
            {isAnomaly ? `High Risk (${riskPct}%)` : `Normal (${riskPct}%)`}
          </span>
        </div>
        {tracePattern && (() => {
          const patternMeta = {
            cycle: { label: 'Cycle', color: '#f97316', desc: 'Circular fund flow detected' },
            fan_out: { label: 'Fan-Out', color: '#ef4444', desc: 'Funds dispersed to many wallets' },
            fan_in: { label: 'Fan-In', color: '#a855f7', desc: 'Funds consolidated from many wallets' },
            peeling_chain: { label: 'Peeling Chain', color: '#eab308', desc: 'Sequential layering / mixing' },
            direct_transfer: { label: 'Direct Transfer', color: '#22c55e', desc: 'Simple point-to-point flow' },
          };
          const meta = patternMeta[tracePattern.pattern] || { label: tracePattern.pattern, color: '#94a3b8', desc: '' };
          return (
            <div style={{ fontSize: '0.8rem', color: '#94a3b8', display: 'flex', gap: '8px', marginTop: '6px', alignItems: 'center', flexWrap: 'wrap' }}>
              <span style={{ background: meta.color + '22', color: meta.color, border: `1px solid ${meta.color}55`, borderRadius: '4px', padding: '2px 8px', fontWeight: 600 }}>
                {meta.label}
              </span>
              <span style={{ color: '#64748b', fontStyle: 'italic' }}>{meta.desc}</span>
              <span style={{ marginLeft: 'auto' }}>Hops: <strong style={{ color: '#cbd5e1' }}>{tracePattern.hop_count}</strong></span>
            </div>
          );
        })()}

      </div>

      <div className="tabs">
        <div className={`tab ${activeTab === 'SAGE' ? 'active' : ''}`} onClick={() => setActiveTab('SAGE')}>Phase 1: GNN</div>
        <div className={`tab ${activeTab === 'ISOLATION' ? 'active' : ''}`} onClick={() => setActiveTab('ISOLATION')}>Phase 2: Risk</div>
        <div className={`tab ${activeTab === 'EXPLAIN' ? 'active' : ''}`} onClick={() => setActiveTab('EXPLAIN')}>Phase 3: SHAP</div>
        <div className={`tab ${activeTab === 'LLM' ? 'active' : ''}`} onClick={() => setActiveTab('LLM')}>Phase 4: LLM</div>
      </div>

      {activeTab === 'SAGE' && (
        <div className="panel-card">
          <h3><Network size={16} style={{ display: 'inline', marginRight: '8px' }} /> GraphSAGE Features</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Network topology & structural embeddings.</p>
          
          <div className="metric-row">
            <span className="metric-label">Total Connections (Network)</span>
            <span><strong>{selectedNode.degree || (selectedNode.in_degree + selectedNode.out_degree) || 0}</strong></span>
          </div>
          <div className="metric-row">
            <span className="metric-label">In-Degree / Out-Degree</span>
            <span>{selectedNode.in_degree || 0} in / {selectedNode.out_degree || 0} out</span>
          </div>
          {selectedNode.visible_degree !== undefined && selectedNode.visible_degree !== selectedNode.degree && (
            <div className="metric-row" style={{ opacity: 0.8, fontSize: '0.85rem' }}>
              <span className="metric-label">Visible in View</span>
              <span>{selectedNode.visible_in_degree || 0} in / {selectedNode.visible_out_degree || 0} out ({selectedNode.visible_degree} total)</span>
            </div>
          )}
          <div className="metric-row">
            <span className="metric-label">PageRank Centrality</span>
            <span>{(selectedNode.pagerank || 0).toFixed(6)}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Clustering Coefficient</span>
            <span>{(selectedNode.clustering_coefficient || 0).toFixed(4)}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Betweenness Centrality</span>
            <span>{(selectedNode.betweenness_centrality || 0).toFixed(6)}</span>
          </div>
          <div className="metric-row">
            <span className="metric-label">Community ID</span>
            <span>{selectedNode.community_id || '0'}</span>
          </div>
        </div>
      )}

      {activeTab === 'ISOLATION' && (
        <div className="panel-card" style={{ borderLeft: isAnomaly ? '4px solid #ef4444' : '4px solid #3b82f6' }}>
          <h3><Activity size={16} style={{ display: 'inline', marginRight: '8px' }} /> ML Risk Scoring</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Ensemble LightGBM/XGBoost & Isolation Forest scores.</p>
          
          <div className="metric-row">
            <span className="metric-label">Risk Probability</span>
            <span style={{ fontWeight: 'bold', color: isAnomaly ? '#ef4444' : '#3b82f6', fontSize: '1.1rem' }}>
              {riskPct}%
            </span>
          </div>
          
          <div className="metric-row">
            <span className="metric-label">Anomaly Classification</span>
            <span style={{ fontWeight: '600', color: isAnomaly ? '#ef4444' : '#10b981' }}>
              {isAnomaly ? 'FLAGGED ANOMALY' : 'PASS (NORMAL)'}
            </span>
          </div>

          <div className="metric-row">
            <span className="metric-label">Total Sent Sats</span>
            <span>{(selectedNode.total_sent_sats || 0).toLocaleString()} sats</span>
          </div>

          <div className="metric-row">
            <span className="metric-label">Total Received Sats</span>
            <span>{(selectedNode.total_received_sats || 0).toLocaleString()} sats</span>
          </div>

          <div className="metric-row">
            <span className="metric-label">Transaction Count</span>
            <span>{selectedNode.tx_count || selectedNode.degree || 0}</span>
          </div>
        </div>
      )}

      {activeTab === 'EXPLAIN' && (
        <div className="panel-card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <h3><Target size={16} style={{ display: 'inline', marginRight: '8px' }} /> Dynamic SHAP Explainability</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Feature impact contributions for this specific wallet.</p>
          
          {isLoadingShap ? (
             <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
                 <Loader className="spinner" size={24} color="#3b82f6" />
             </div>
          ) : shapData && shapData.length > 0 ? (
             <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '12px' }}>
                {shapData.slice(0, 5).map((f, idx) => {
                    const isIncrease = f.direction === 'increases_risk';
                    return (
                        <div key={idx} style={{ background: 'rgba(0,0,0,0.3)', padding: '12px', borderRadius: '8px', borderLeft: isIncrease ? '3px solid #ef4444' : '3px solid #10b981' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                               <span style={{ fontSize: '0.85rem', fontWeight: 'bold', color: '#f1f5f9' }}>{f.feature || f.feature_name}</span>
                               <span style={{ fontSize: '0.75rem', fontWeight: '600', color: isIncrease ? '#ef4444' : '#10b981' }}>
                                  {isIncrease ? 'Increases Risk' : 'Decreases Risk'}
                               </span>
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#94a3b8', marginBottom: '4px' }}>
                               {f.description || `Impact on risk: ${f.shap_value}`}
                            </div>
                            <div style={{ fontSize: '0.75rem', color: '#64748b' }}>
                               Value: <strong>{Number(f.raw_value || f.feature_value || 0).toLocaleString()}</strong> | SHAP: <strong>{Number(f.shap_value || 0).toFixed(4)}</strong>
                            </div>
                        </div>
                    );
                })}
             </div>
          ) : (
             <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>No SHAP factors returned for this node.</p>
          )}
        </div>
      )}

      {activeTab === 'LLM' && (
        <div className="panel-card" style={{ borderLeft: '4px solid #a855f7' }}>
          <h3><MessageSquare size={16} style={{ display: 'inline', marginRight: '8px' }} /> LLM Analyst Explanation</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Natural language forensic summary driven by local LLM.</p>
          
          {!llmExplainData && !isLoadingLLM && (
            <button className="btn" onClick={handleFetchLLM} style={{ width: '100%', background: '#a855f7', color: 'white', border: 'none', padding: '10px', borderRadius: '6px' }}>
              <Zap size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
              Generate Explanation
            </button>
          )}

          {isLoadingLLM && (
             <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: '20px' }}>
                 <Loader className="spinner" size={28} color="#a855f7" />
                 <p style={{ marginTop: '12px', color: '#cbd5e1', fontSize: '0.9rem' }}>Thinking... This may take up to 20-30 seconds depending on the model.</p>
             </div>
          )}

          {llmExplainData && !llmExplainData.error && (
             <div style={{ marginTop: '12px', background: 'rgba(0,0,0,0.3)', padding: '16px', borderRadius: '8px', borderLeft: '3px solid #a855f7' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px', borderBottom: '1px solid #334155', paddingBottom: '8px' }}>
                   <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Model: <strong>{llmExplainData.model_used}</strong></span>
                   {llmExplainData.cached && <span style={{ fontSize: '0.75rem', background: '#334155', padding: '2px 6px', borderRadius: '4px' }}>Cached</span>}
                </div>
                <div style={{ fontSize: '0.9rem', color: '#f1f5f9', lineHeight: '1.6', whiteSpace: 'pre-wrap' }}>
                   {llmExplainData.explanation}
                </div>
             </div>
          )}
          
          {llmExplainData && llmExplainData.error && (
             <div style={{ marginTop: '12px', background: 'rgba(239,68,68,0.1)', padding: '12px', borderRadius: '8px', borderLeft: '3px solid #ef4444' }}>
                <p style={{ color: '#fca5a5', fontSize: '0.9rem' }}>{llmExplainData.message}</p>
             </div>
          )}
        </div>
      )}

      <button className="btn trace" onClick={handleTraceClick} disabled={isTracing}>
        <Zap size={16} style={{ display: 'inline', marginRight: '6px', verticalAlign: 'text-bottom' }} />
        {isTracing ? 'Tracing Funds...' : 'Dynamic Trace Flow'}
      </button>
    </div>
  );
}
