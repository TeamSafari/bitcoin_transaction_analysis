import React, { useState } from 'react';
import { ShieldAlert, ShieldCheck, Activity, Network, Target } from 'lucide-react';

export default function Sidebar({ selectedNode }) {
  const [activeTab, setActiveTab] = useState('SAGE'); // SAGE, ISOLATION, EXPLAIN

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
        <div className="metric-row"><span className="metric-label">Status</span><span className={`badge ${isAnomaly ? 'danger' : 'safe'}`}>{isAnomaly ? 'High Risk Anomaly' : 'Normal Activity'}</span></div>
      </div>

      <div className="tabs">
        <div className={`tab ${activeTab === 'SAGE' ? 'active' : ''}`} onClick={() => setActiveTab('SAGE')}>Phase 1: GNN</div>
        <div className={`tab ${activeTab === 'ISOLATION' ? 'active' : ''}`} onClick={() => setActiveTab('ISOLATION')}>Phase 2: Anomaly</div>
        <div className={`tab ${activeTab === 'EXPLAIN' ? 'active' : ''}`} onClick={() => setActiveTab('EXPLAIN')}>Phase 3: Explain</div>
      </div>

      {activeTab === 'SAGE' && (
        <div className="panel-card">
          <h3><Network size={16} style={{ display: 'inline', marginRight: '8px' }} /> GraphSAGE Features</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Features extracted from network topology and message passing.</p>
          <div className="metric-row"><span className="metric-label">Degree (Total Conns)</span><span>{selectedNode.degree}</span></div>
          <div className="metric-row"><span className="metric-label">PageRank Centrality</span><span>{selectedNode.pagerank.toFixed(6)}</span></div>
          <div className="metric-row"><span className="metric-label">Community ID</span><span>{selectedNode.community_id}</span></div>
        </div>
      )}

      {activeTab === 'ISOLATION' && (
        <div className="panel-card" style={{ borderLeft: isAnomaly ? '4px solid #ef4444' : '4px solid #3b82f6' }}>
          <h3><Activity size={16} style={{ display: 'inline', marginRight: '8px' }} /> Isolation Forest</h3>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>Detects statistical outliers within the GraphSAGE clusters.</p>
          <div className="metric-row"><span className="metric-label">Anomaly Score</span><span style={{ fontWeight: 'bold' }}>{selectedNode.anomaly_score.toFixed(4)}</span></div>
          <div className="metric-row"><span className="metric-label">Threshold Result</span><span>{isAnomaly ? 'Flagged' : 'Pass'}</span></div>
        </div>
      )}

      {activeTab === 'EXPLAIN' && (
        <div className="panel-card" style={{ borderLeft: '4px solid #f59e0b' }}>
          <h3><Target size={16} style={{ display: 'inline', marginRight: '8px' }} /> Explainability (XAI)</h3>
          {isAnomaly ? (
            <p style={{ marginTop: '12px', fontSize: '0.9rem', lineHeight: '1.5' }}>
              <strong>Placeholder Rule:</strong> This wallet deviates significantly from Community {selectedNode.community_id}. 
              It has a high PageRank ({selectedNode.pagerank.toFixed(5)}) relative to its low transaction count, suggesting it acts as an intermediary in a layered laundering scheme.
              <br/><br/>
              <em>Note: Once `feature_importance.json` is exported from the backend, real SHAP values will appear here.</em>
            </p>
          ) : (
            <p style={{ fontSize: '0.9rem' }}>No significant anomalies detected to explain.</p>
          )}
        </div>
      )}

      <button className="btn trace" onClick={() => window.traceFunds()}>Trace Fund Flow (Demo)</button>
    </div>
  );
}
