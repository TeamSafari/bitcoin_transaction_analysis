import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Activity, Network, Target, ChevronRight } from 'lucide-react';

export default function Sidebar({ selectedNode, onTrace }) {
  const [activeTab, setActiveTab] = useState('SAGE'); 

  if (!selectedNode) {
    return (
      <div className="sidebar">
        <h2><Target color="#3b82f6" /> XAI Analyst View</h2>
        <p style={{ color: 'var(--text-muted)' }}>Use the search bar to extract an Ego-Graph, or select a wallet to view its multi-stage analysis.</p>
      </div>
    );
  }

  const isAnomaly = selectedNode.is_anomaly;

  return (
    <div className="sidebar">
      <h2>{isAnomaly ? <ShieldAlert color="#ef4444" /> : <ShieldCheck color="#3b82f6" />} Entity Profile</h2>
      
      <div className="panel-card">
        <h3>{selectedNode.id}</h3>
        <div className="metric-row"><span className="metric-label">Status</span><span className={`badge ${isAnomaly ? 'danger' : 'safe'}`}>{isAnomaly ? 'High Risk' : 'Normal'}</span></div>
      </div>

      <div className="tabs">
        <div className={`tab ${activeTab === 'SAGE' ? 'active' : ''}`} onClick={() => setActiveTab('SAGE')}>Phase 1: GNN</div>
        <div className={`tab ${activeTab === 'ISOLATION' ? 'active' : ''}`} onClick={() => setActiveTab('ISOLATION')}>Phase 2: Risk</div>
        <div className={`tab ${activeTab === 'EXPLAIN' ? 'active' : ''}`} onClick={() => setActiveTab('EXPLAIN')}>Phase 3: Explain</div>
      </div>

      {activeTab === 'SAGE' && (
        <div className="panel-card">
          <h3><Network size={16} style={{ display: 'inline', marginRight: '8px' }} /> GraphSAGE Features</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Features extracted from network topology.</p>
          <div className="metric-row"><span className="metric-label">Degree (Total Conns)</span><span>{selectedNode.degree}</span></div>
          <div className="metric-row"><span className="metric-label">PageRank Centrality</span><span>{selectedNode.pagerank.toFixed(6)}</span></div>
          <div className="metric-row"><span className="metric-label">Community ID</span><span>{selectedNode.community_id}</span></div>
        </div>
      )}

      {activeTab === 'ISOLATION' && (
        <div className="panel-card" style={{ borderLeft: isAnomaly ? '4px solid #ef4444' : '4px solid #3b82f6' }}>
          <h3><Activity size={16} style={{ display: 'inline', marginRight: '8px' }} /> Fusion Risk Model</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Final prediction from the fused ML pipeline.</p>
          <div className="metric-row"><span className="metric-label">Risk Probability</span><span style={{ fontWeight: 'bold' }}>{(selectedNode.risk_probability * 100).toFixed(2)}%</span></div>
          <div className="metric-row"><span className="metric-label">Threshold Result</span><span>{isAnomaly ? 'Flagged' : 'Pass'}</span></div>
        </div>
      )}

      {activeTab === 'EXPLAIN' && (
        <div className="panel-card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <h3><Target size={16} style={{ display: 'inline', marginRight: '8px' }} /> Explainability (XAI)</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Dynamic SHAP feature contributions for this specific wallet.</p>
          
          {selectedNode.shap_features && selectedNode.shap_features.length > 0 ? (
             <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: '12px' }}>
                {selectedNode.shap_features.slice(0, 3).map((f, idx) => (
                    <div key={idx} style={{ background: 'rgba(0,0,0,0.2)', padding: '10px', borderRadius: '8px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                           <span style={{ fontSize: '0.85rem', fontWeight: 'bold' }}>{f.feature}</span>
                           <span style={{ fontSize: '0.8rem', color: f.direction === 'increases_risk' ? '#ef4444' : '#10b981' }}>
                              {f.direction === 'increases_risk' ? 'Increases Risk' : 'Decreases Risk'}
                           </span>
                        </div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Value: {Number(f.feature_value).toFixed(4)} | SHAP: {Number(f.shap_value).toFixed(4)}</div>
                    </div>
                ))}
             </div>
          ) : (
            <p style={{ fontSize: '0.9rem' }}>No SHAP data available for this node.</p>
          )}
        </div>
      )}

      <button className="btn trace" onClick={() => onTrace(selectedNode.id)}>Dynamic Trace Flow</button>
    </div>
  );
}
