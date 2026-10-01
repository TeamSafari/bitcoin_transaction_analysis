import React, { useState, useMemo } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
  AlertTriangle, Users, ShieldAlert, History, Network,
  Zap, ArrowUpRight, TrendingUp, CheckCircle2, Copy,
  Check, Layers, Cpu, ArrowRight, ShieldCheck, BarChart3,
  ExternalLink, Sparkles
} from 'lucide-react';
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';

// ─── Helper: derive pattern from features ─────────────────────────────────────
function detectPattern(feats) {
  const inDeg = feats.in_degree || 0;
  const outDeg = feats.out_degree || 0;
  if (inDeg === 0 && outDeg === 0) return null;
  if (outDeg > inDeg + 3) return 'Fan-out (Scattering)';
  if (inDeg > outDeg + 3) return 'Fan-in (Gathering)';
  if (inDeg > 0 && outDeg > 0 && Math.abs(inDeg - outDeg) <= 3) return 'Pass-through (Peeling)';
  return 'Direct Transfer';
}

// ─── Helper: format satoshis to readable string ──────────────────────────────
function formatSatoshis(sats) {
  if (!sats || sats <= 0) return '0 SATS';
  if (sats >= 1e8) return `${(sats / 1e8).toFixed(2)} BTC`;
  if (sats >= 1e6) return `${(sats / 1e6).toFixed(1)}M SATS`;
  if (sats >= 1e3) return `${(sats / 1e3).toFixed(1)}k SATS`;
  return `${Math.round(sats)} SATS`;
}

export default function Dashboard() {
  const {
    globalJobId, setGlobalJobId, jobsList,
    cachedGraphElements, cachedAlerts,
    dataLoading, dataReady,
  } = useOutletContext();
  const navigate = useNavigate();

  const [activeQueueTab, setActiveQueueTab] = useState('ALL');
  const [copiedId, setCopiedId] = useState(null);

  const completedJobs = (jobsList || []).filter(j => j.status === 'completed');
  const elements = cachedGraphElements || [];
  const alerts = cachedAlerts || [];

  // Nodes & Edges
  const nodes = useMemo(() => elements.filter(el => el.data && !el.data.source), [elements]);
  const edges = useMemo(() => elements.filter(el => el.data && el.data.source), [elements]);
  const anomalyNodes = useMemo(() => nodes.filter(n => n.data.is_anomaly), [nodes]);
  const cleanNodes = useMemo(() => nodes.filter(n => !n.data.is_anomaly), [nodes]);

  // Node features lookup
  const nodeFeats = useMemo(() => {
    const lookup = {};
    nodes.forEach(n => { lookup[n.data.id] = n.data.wallet_features || {}; });
    return lookup;
  }, [nodes]);

  // Total Satoshis analyzed
  const totalSatoshis = useMemo(() => {
    let sum = 0;
    edges.forEach(e => {
      const w = e.data?.weight || e.data?.amount_sats || 0;
      sum += Number(w) || 0;
    });
    if (sum === 0) {
      nodes.forEach(n => {
        sum += Number(n.data?.wallet_features?.total_sent_sats || 0);
      });
    }
    return sum;
  }, [edges, nodes]);

  // Average degree
  const avgDegree = useMemo(() => {
    if (nodes.length === 0) return 0;
    const totalDeg = nodes.reduce((acc, n) => acc + (n.data?.degree || 0), 0);
    return (totalDeg / nodes.length).toFixed(1);
  }, [nodes]);

  // Topology Pattern Counts
  const patternData = useMemo(() => {
    const counts = {};
    alerts.forEach(a => {
      const feats = nodeFeats[a.wallet_id] || {};
      const pat = detectPattern(feats);
      if (pat) counts[pat] = (counts[pat] || 0) + 1;
    });
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [alerts, nodeFeats]);

  // Multi-Model Detection Breakdown
  const modelMetrics = useMemo(() => {
    let gnn = 0;
    let iso = 0;
    let ae = 0;
    let rules = 0;

    alerts.forEach(a => {
      if (a.risk_prediction === 1) gnn++;
      if ((a.isolation_forest_score || 0) > 0.5) iso++;
      if ((a.autoencoder_score || 0) > 0.5) ae++;
      if ((a.deterministic_score || 0) > 0.5) rules++;
    });

    return [
      { name: 'GraphSAGE GNN', count: gnn, color: '#3b82f6', desc: 'Structural Graph Anomaly' },
      { name: 'Isolation Forest', count: iso, color: '#10b981', desc: 'High-Dimensional Outlier' },
      { name: 'Autoencoder', count: ae, color: '#8b5cf6', desc: 'Reconstruction Loss' },
      { name: 'Rule Engine', count: rules, color: '#f59e0b', desc: 'Velocity & Dispersal Heuristics' },
    ];
  }, [alerts]);

  // Consensus Alerts (flagged by 2 or more models)
  const consensusAlerts = useMemo(() => {
    return alerts.filter(a => {
      let votes = 0;
      if (a.risk_prediction === 1) votes++;
      if ((a.isolation_forest_score || 0) > 0.5) votes++;
      if ((a.autoencoder_score || 0) > 0.5) votes++;
      if ((a.deterministic_score || 0) > 0.5) votes++;
      return votes >= 2;
    });
  }, [alerts]);

  // Risk Tiers
  const criticalAlerts = useMemo(() => alerts.filter(a => (a.risk_probability || 0) >= 0.85 || a.severity === 'CRITICAL'), [alerts]);
  const highAlerts = useMemo(() => alerts.filter(a => ((a.risk_probability || 0) >= 0.70 && (a.risk_probability || 0) < 0.85) || (a.severity === 'HIGH' && (a.risk_probability || 0) < 0.85)), [alerts]);
  const mediumAlerts = useMemo(() => alerts.filter(a => ((a.risk_probability || 0) >= 0.40 && (a.risk_probability || 0) < 0.70) || a.severity === 'MEDIUM'), [alerts]);
  const lowAlerts = useMemo(() => alerts.filter(a => (a.risk_probability || 0) < 0.40 && a.severity !== 'HIGH' && a.severity !== 'CRITICAL' && a.severity !== 'MEDIUM'), [alerts]);

  // Global Average Risk Score
  const avgRiskScore = useMemo(() => {
    if (alerts.length === 0) return 0;
    const sum = alerts.reduce((acc, a) => acc + (a.risk_probability || 0), 0);
    return Math.round((sum / alerts.length) * 100);
  }, [alerts]);

  // Global SHAP Risk Drivers
  const topShapDrivers = useMemo(() => {
    const impactMap = {};
    alerts.forEach(a => {
      const factors = a.top_shap_factors || [];
      factors.forEach(f => {
        const feat = f.feature || 'unknown';
        const formatted = feat.replace(/_/g, ' ');
        const absVal = Math.abs(f.shap_value || f.contribution || f.absolute_shap_value || 0.1);
        if (!impactMap[formatted]) {
          impactMap[formatted] = { name: formatted, count: 0, totalImpact: 0 };
        }
        impactMap[formatted].count += 1;
        impactMap[formatted].totalImpact += absVal;
      });
    });

    return Object.values(impactMap)
      .sort((a, b) => b.totalImpact - a.totalImpact)
      .slice(0, 5)
      .map(item => ({
        name: item.name.length > 20 ? item.name.substring(0, 20) + '…' : item.name,
        fullName: item.name,
        impact: Number((item.totalImpact / Math.max(1, alerts.length)).toFixed(3)),
        frequency: item.count,
      }));
  }, [alerts]);

  // Top Suspicious Queue
  const queueList = useMemo(() => {
    let sourceList = alerts;
    if (activeQueueTab === 'CRITICAL') {
      sourceList = criticalAlerts;
    } else if (activeQueueTab === 'CONSENSUS') {
      sourceList = consensusAlerts;
    }
    return [...sourceList]
      .sort((a, b) => (b.risk_probability || 0) - (a.risk_probability || 0))
      .slice(0, 8);
  }, [alerts, activeQueueTab, criticalAlerts, consensusAlerts]);

  const handleCopy = (id) => {
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const PATTERN_COLORS = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6'];

  // ── No data state ──────────────────────────────────────────
  if (!dataReady && !dataLoading && !globalJobId) {
    return (
      <div style={{ padding: '32px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: 'rgba(37,99,235,0.08)', padding: '24px', borderRadius: '50%', marginBottom: 20 }}>
          <Network size={54} color="var(--accent)" />
        </div>
        <h2 style={{ fontWeight: 700, fontSize: '1.4rem', marginBottom: 8, color: 'var(--text-primary)' }}>
          Forensics Intelligence Dashboard
        </h2>
        <p style={{ color: 'var(--text-muted)', maxWidth: 460, textAlign: 'center', lineHeight: 1.6, fontSize: '0.9rem' }}>
          No active analysis dataset loaded. Upload transaction files in the <strong>Network Graph</strong> or select a pre-processed job to begin forensic investigation.
        </p>
        <button
          className="action-btn primary"
          style={{ width: 'auto', padding: '12px 28px', marginTop: 24, fontSize: '0.92rem', borderRadius: '10px' }}
          onClick={() => navigate('/graph')}
        >
          <Network size={16} /> Go to Network Graph
        </button>
      </div>
    );
  }

  return (
    <div style={{ padding: '28px 32px', height: '100%', overflowY: 'auto' }}>
      
      {/* ══ TOP HEADER ══════════════════════════════════════════════ */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '24px', flexWrap: 'wrap', gap: '16px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <h1 style={{ fontSize: '1.55rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              Forensic Intelligence Dashboard
            </h1>
            {globalJobId && (
              <span style={{
                background: 'rgba(37,99,235,0.1)',
                color: 'var(--accent)',
                padding: '3px 10px',
                borderRadius: '6px',
                fontSize: '0.78rem',
                fontWeight: 600,
                border: '1px solid rgba(37,99,235,0.2)'
              }}>
                Job: {globalJobId}
              </span>
            )}
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', marginTop: 4 }}>
            Multi-model graph anomaly detection, risk probability scoring, and topological AML heuristics
          </p>
        </div>

        {/* Quick controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {completedJobs.length > 0 && (
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: '10px', padding: '8px 12px', boxShadow: '0 2px 6px rgba(0,0,0,0.03)'
            }}>
              <History size={15} color="var(--text-muted)" />
              <select
                value={globalJobId || ''}
                onChange={(e) => setGlobalJobId(e.target.value || null)}
                style={{
                  background: 'transparent', color: 'var(--text-primary)',
                  border: 'none', fontSize: '0.83rem', outline: 'none',
                  cursor: 'pointer', fontWeight: 500
                }}
              >
                <option value="">Select Job</option>
                {completedJobs.map(j => (
                  <option key={j.job_id} value={j.job_id}>
                    {j.job_id}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button
            className="action-btn outline"
            style={{ width: 'auto', padding: '8px 14px', borderRadius: '10px', fontSize: '0.82rem' }}
            onClick={() => navigate('/graph')}
          >
            <Network size={14} /> Network Graph
          </button>

          <button
            className="action-btn outline"
            style={{ width: 'auto', padding: '8px 14px', borderRadius: '10px', fontSize: '0.82rem' }}
            onClick={() => navigate('/records')}
          >
            <Layers size={14} /> View Records
          </button>
        </div>
      </div>

      {/* ══ KPI METRICS ROW ═════════════════════════════════════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '16px', marginBottom: '28px' }}>
        
        {/* Total Wallets */}
        <div className="data-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Monitored Wallets
            </span>
            <div style={{ background: 'rgba(59,130,246,0.1)', padding: '8px', borderRadius: '8px' }}>
              <Users size={18} color="var(--accent)" />
            </div>
          </div>
          <div style={{ fontSize: '1.9rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>
            {nodes.length}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: 10, fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
            <span style={{ color: 'var(--success)', fontWeight: 600 }}>{cleanNodes.length} normal</span>
            <span>•</span>
            <span style={{ color: 'var(--danger)', fontWeight: 600 }}>{anomalyNodes.length} flagged</span>
          </div>
        </div>

        {/* Total Transaction Flow */}
        <div className="data-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Transaction Volume
            </span>
            <div style={{ background: 'rgba(16,185,129,0.1)', padding: '8px', borderRadius: '8px' }}>
              <TrendingUp size={18} color="var(--success)" />
            </div>
          </div>
          <div style={{ fontSize: '1.9rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>
            {edges.length}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: 10, fontSize: '0.76rem', color: 'var(--text-muted)' }}>
            <span>Flow:</span>
            <strong style={{ color: 'var(--text-primary)' }}>{formatSatoshis(totalSatoshis)}</strong>
          </div>
        </div>

        {/* High & Critical Alerts */}
        <div className="data-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              High-Risk Alerts
            </span>
            <div style={{ background: 'rgba(239,68,68,0.1)', padding: '8px', borderRadius: '8px' }}>
              <ShieldAlert size={18} color="var(--danger)" />
            </div>
          </div>
          <div style={{ fontSize: '1.9rem', fontWeight: 700, color: 'var(--danger)', lineHeight: 1 }}>
            {criticalAlerts.length + highAlerts.length}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: 10, fontSize: '0.76rem', color: 'var(--text-muted)' }}>
            <span>Critical: <strong style={{ color: '#ef4444' }}>{criticalAlerts.length}</strong></span>
            <span>•</span>
            <span>High: <strong style={{ color: '#f97316' }}>{highAlerts.length}</strong></span>
          </div>
        </div>

        {/* Multi-Model Consensus */}
        <div className="data-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Model Consensus
            </span>
            <div style={{ background: 'rgba(245,158,11,0.1)', padding: '8px', borderRadius: '8px' }}>
              <Zap size={18} color="var(--warning)" />
            </div>
          </div>
          <div style={{ fontSize: '1.9rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>
            {consensusAlerts.length}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: 10, fontSize: '0.76rem', color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--warning)', fontWeight: 600 }}>Multi-engine agreement</span>
          </div>
        </div>

        {/* Topology Density */}
        <div className="data-card" style={{ padding: '20px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Avg Connections
            </span>
            <div style={{ background: 'rgba(139,92,246,0.1)', padding: '8px', borderRadius: '8px' }}>
              <Network size={18} color="var(--purple)" />
            </div>
          </div>
          <div style={{ fontSize: '1.9rem', fontWeight: 700, color: 'var(--text-primary)', lineHeight: 1 }}>
            {avgDegree}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: 10, fontSize: '0.76rem', color: 'var(--text-muted)' }}>
            <span>Network Density Metric</span>
          </div>
        </div>

      </div>

      {/* ══ ROW 1: TOPOLOGY & MULTI-MODEL COMPARISON ════════════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        
        {/* Topology Patterns Donut */}
        <div className="data-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
                Detected Topology Patterns
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Structural distribution across transaction sequences
              </p>
            </div>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent)', background: 'rgba(37,99,235,0.08)', padding: '3px 8px', borderRadius: '6px' }}>
              {patternData.reduce((acc, p) => acc + p.value, 0)} Patterns
            </span>
          </div>

          <div style={{ height: 260, position: 'relative' }}>
            {patternData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={patternData}
                    cx="50%"
                    cy="50%"
                    innerRadius={62}
                    outerRadius={92}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {patternData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={PATTERN_COLORS[index % PATTERN_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: 'var(--bg-card)',
                      border: '1px solid var(--border)',
                      borderRadius: '8px',
                      boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                      fontSize: '0.8rem'
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--text-muted)', fontSize: '0.84rem' }}>
                No topology patterns detected
              </div>
            )}
          </div>

          {/* Pattern Legend Chips */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px', marginTop: '12px' }}>
            {patternData.map((p, idx) => (
              <div
                key={p.name}
                style={{
                  display: 'flex', alignItems: 'center', gap: '6px',
                  background: 'var(--bg-deep)', padding: '5px 10px',
                  borderRadius: '6px', border: '1px solid var(--border)',
                  fontSize: '0.75rem'
                }}
              >
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: PATTERN_COLORS[idx % PATTERN_COLORS.length] }} />
                <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>{p.name}:</span>
                <strong style={{ color: 'var(--text-primary)' }}>{p.value}</strong>
              </div>
            ))}
          </div>
        </div>

        {/* Multi-Model Detection Breakdown */}
        <div className="data-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
                Multi-Model Detection Consensus
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Flagged count per AI/ML model and deterministic rule engine
              </p>
            </div>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--purple)', background: 'rgba(139,92,246,0.08)', padding: '3px 8px', borderRadius: '6px' }}>
              4 Detection Engines
            </span>
          </div>

          <div style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={modelMetrics} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="name" stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={11} tickLine={false} />
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-card)',
                    border: '1px solid var(--border)',
                    borderRadius: '8px',
                    boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                    fontSize: '0.8rem'
                  }}
                  formatter={(val, name, item) => [`${val} Wallets Flagged`, item.payload.desc]}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                  {modelMetrics.map((entry, idx) => (
                    <Cell key={`bar-${idx}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px', marginTop: '12px' }}>
            {modelMetrics.map(m => (
              <div key={m.name} style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                <div style={{ width: 8, height: 8, borderRadius: '2px', background: m.color }} />
                <span style={{ color: 'var(--text-secondary)' }}>{m.name}:</span>
                <strong style={{ color: 'var(--text-primary)' }}>{m.count}</strong>
              </div>
            ))}
          </div>
        </div>

      </div>

      {/* ══ ROW 2: RISK DISTRIBUTION & SHAP EXPLAINABILITY ═══════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '20px', marginBottom: '24px' }}>
        
        {/* Risk Score Distribution Tiers */}
        <div className="data-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
                Risk Severity Breakdown
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Entities categorized across 4 threat probability tiers
              </p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>Average Risk:</span>
              <strong style={{ marginLeft: 6, fontSize: '0.85rem', color: 'var(--accent)' }}>{avgRiskScore}%</strong>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: 10 }}>
            {[
              { label: 'Critical Risk (85% - 100%)', count: criticalAlerts.length, color: '#ef4444', bg: 'rgba(239,68,68,0.1)' },
              { label: 'High Risk (70% - 84%)', count: highAlerts.length, color: '#f97316', bg: 'rgba(249,115,22,0.1)' },
              { label: 'Medium Risk (40% - 69%)', count: mediumAlerts.length, color: '#f59e0b', bg: 'rgba(245,158,11,0.1)' },
              { label: 'Low / Safe (< 40%)', count: lowAlerts.length, color: '#10b981', bg: 'rgba(16,185,129,0.1)' },
            ].map(tier => {
              const pct = alerts.length > 0 ? ((tier.count / alerts.length) * 100).toFixed(1) : 0;
              return (
                <div key={tier.label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 4 }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>{tier.label}</span>
                    <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{tier.count} <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({pct}%)</span></span>
                  </div>
                  <div style={{ height: 8, background: 'var(--bg-deep)', borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      style={{
                        height: '100%',
                        width: `${pct}%`,
                        background: tier.color,
                        borderRadius: 4,
                        transition: 'width 0.4s ease'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div style={{ marginTop: 20, padding: '12px 14px', background: 'var(--bg-deep)', borderRadius: '8px', border: '1px solid var(--border)', fontSize: '0.76rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: 8 }}>
            <Sparkles size={14} color="var(--accent)" style={{ flexShrink: 0 }} />
            <span>Entities with score &gt; 70% require heightened review under automated AML routing policy.</span>
          </div>
        </div>

        {/* Global SHAP Feature Drivers */}
        <div className="data-card" style={{ padding: '22px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <div>
              <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
                Primary SHAP Risk Drivers
              </h3>
              <p style={{ margin: '2px 0 0', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                Top features contributing to ML anomaly classification
              </p>
            </div>
            <span style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--accent)', background: 'rgba(37,99,235,0.08)', padding: '3px 8px', borderRadius: '6px' }}>
              Explainability
            </span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginTop: 8 }}>
            {topShapDrivers.length > 0 ? (
              topShapDrivers.map((driver, index) => {
                const maxImpact = topShapDrivers[0]?.impact || 1;
                const barWidth = Math.min(100, Math.max(15, (driver.impact / maxImpact) * 100));
                return (
                  <div key={driver.name} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                      <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>
                        #{index + 1} {driver.fullName}
                      </span>
                      <span style={{ color: 'var(--danger)', fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>
                        +{driver.impact} impact <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({driver.frequency}x)</span>
                      </span>
                    </div>
                    <div style={{ height: 6, background: 'var(--bg-deep)', borderRadius: 3, overflow: 'hidden' }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${barWidth}%`,
                          background: 'linear-gradient(90deg, #ef4444, #f97316)',
                          borderRadius: 3
                        }}
                      />
                    </div>
                  </div>
                );
              })
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: 180, color: 'var(--text-muted)', fontSize: '0.84rem' }}>
                SHAP factors loading or unavailable for this dataset
              </div>
            )}
          </div>

          <div style={{ marginTop: 18, fontSize: '0.74rem', color: 'var(--text-muted)' }}>
            * Values reflect average absolute Shapley impact across high-dimensional graph features.
          </div>
        </div>

      </div>

      {/* ══ ROW 3: PRIORITY FORENSICS QUEUE (TOP HIGH-RISK WALLETS) ══ */}
      <div className="data-card" style={{ padding: '22px', marginBottom: '28px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '18px', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.96rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
              Priority Action Queue — Highest-Risk Entities
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '0.77rem', color: 'var(--text-muted)' }}>
              Actionable triage queue with 1-click network graph investigation
            </p>
          </div>

          {/* Filter tabs */}
          <div style={{ display: 'flex', gap: '6px', background: 'var(--bg-deep)', padding: '3px', borderRadius: '8px', border: '1px solid var(--border)' }}>
            {[
              { id: 'ALL', label: `All High Risk (${criticalAlerts.length + highAlerts.length})` },
              { id: 'CRITICAL', label: `Critical Only (${criticalAlerts.length})` },
              { id: 'CONSENSUS', label: `Model Consensus (${consensusAlerts.length})` },
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveQueueTab(tab.id)}
                style={{
                  background: activeQueueTab === tab.id ? 'var(--bg-card)' : 'transparent',
                  color: activeQueueTab === tab.id ? 'var(--text-primary)' : 'var(--text-muted)',
                  border: 'none',
                  borderRadius: '6px',
                  padding: '5px 12px',
                  fontSize: '0.76rem',
                  fontWeight: activeQueueTab === tab.id ? 600 : 500,
                  cursor: 'pointer',
                  boxShadow: activeQueueTab === tab.id ? '0 1px 3px rgba(0,0,0,0.08)' : 'none',
                  transition: 'all 0.15s ease'
                }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Table */}
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.82rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Rank & Wallet</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Risk Score</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Severity</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Pattern</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Model Votes</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Primary Driver</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase', textAlign: 'right' }}>Action</th>
              </tr>
            </thead>
            <tbody>
              {queueList.length > 0 ? (
                queueList.map((item, index) => {
                  const riskPct = Math.round((item.risk_probability || 0) * 100);
                  const isCrit = riskPct >= 85 || item.severity === 'CRITICAL';
                  const feats = nodeFeats[item.wallet_id] || {};
                  const pattern = detectPattern(feats) || 'Direct Transfer';
                  const primaryDriver = item.top_shap_factors?.[0]?.feature?.replace(/_/g, ' ') || 'High Out-Degree Flow';

                  let votes = 0;
                  if (item.risk_prediction === 1) votes++;
                  if ((item.isolation_forest_score || 0) > 0.5) votes++;
                  if ((item.autoencoder_score || 0) > 0.5) votes++;
                  if ((item.deterministic_score || 0) > 0.5) votes++;

                  return (
                    <tr
                      key={item.wallet_id}
                      style={{
                        borderBottom: '1px solid var(--border)',
                        transition: 'background 0.12s',
                      }}
                      onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                      onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                    >
                      {/* Wallet */}
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 700, minWidth: 20 }}>
                            #{index + 1}
                          </span>
                          <span style={{ fontFamily: 'monospace', fontWeight: 600, color: 'var(--text-primary)' }}>
                            {item.wallet_id.length > 14 ? item.wallet_id.substring(0, 14) + '…' : item.wallet_id}
                          </span>
                          <button
                            onClick={() => handleCopy(item.wallet_id)}
                            title="Copy Wallet ID"
                            style={{
                              background: 'none', border: 'none', cursor: 'pointer',
                              color: copiedId === item.wallet_id ? 'var(--success)' : 'var(--text-muted)',
                              padding: 2, display: 'flex', alignItems: 'center'
                            }}
                          >
                            {copiedId === item.wallet_id ? <Check size={12} /> : <Copy size={12} />}
                          </button>
                        </div>
                      </td>

                      {/* Risk Score */}
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontWeight: 700, color: isCrit ? '#ef4444' : '#f97316', minWidth: 38 }}>
                            {riskPct}%
                          </span>
                          <div style={{ width: 60, height: 6, background: 'var(--bg-deep)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ height: '100%', width: `${riskPct}%`, background: isCrit ? '#ef4444' : '#f97316' }} />
                          </div>
                        </div>
                      </td>

                      {/* Severity Badge */}
                      <td style={{ padding: '12px' }}>
                        <span style={{
                          padding: '3px 8px', borderRadius: '5px', fontSize: '0.73rem', fontWeight: 600,
                          background: isCrit ? 'rgba(239,68,68,0.1)' : 'rgba(249,115,22,0.1)',
                          color: isCrit ? '#ef4444' : '#f97316',
                          border: `1px solid ${isCrit ? 'rgba(239,68,68,0.25)' : 'rgba(249,115,22,0.25)'}`
                        }}>
                          {item.severity || (isCrit ? 'CRITICAL' : 'HIGH')}
                        </span>
                      </td>

                      {/* Pattern */}
                      <td style={{ padding: '12px', color: 'var(--text-secondary)' }}>
                        <span style={{
                          background: 'var(--bg-deep)', padding: '3px 8px', borderRadius: '4px',
                          border: '1px solid var(--border)', fontSize: '0.74rem', fontWeight: 500
                        }}>
                          {pattern}
                        </span>
                      </td>

                      {/* Model Agreement */}
                      <td style={{ padding: '12px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <span style={{
                            fontSize: '0.72rem', fontWeight: 600, padding: '2px 6px', borderRadius: '4px',
                            background: votes >= 3 ? 'rgba(239,68,68,0.1)' : 'rgba(245,158,11,0.1)',
                            color: votes >= 3 ? '#ef4444' : '#f59e0b'
                          }}>
                            {votes}/4 Models
                          </span>
                        </div>
                      </td>

                      {/* Driver */}
                      <td style={{ padding: '12px', color: 'var(--text-secondary)', fontSize: '0.78rem' }}>
                        {primaryDriver}
                      </td>

                      {/* Action */}
                      <td style={{ padding: '12px', textAlign: 'right' }}>
                        <button
                          className="action-btn outline"
                          style={{
                            width: 'auto', display: 'inline-flex', padding: '5px 10px',
                            fontSize: '0.74rem', borderRadius: '6px'
                          }}
                          onClick={() => navigate('/graph')}
                          title="Open in Network Graph"
                        >
                          <Network size={12} /> Graph
                        </button>
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={7} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                    No alerts match the selected priority filter
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
