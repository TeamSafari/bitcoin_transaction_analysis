import React, { useState, useEffect, useMemo } from 'react';
import { useOutletContext, useNavigate } from 'react-router-dom';
import {
  History, Network, Search, X, Copy, Check, ExternalLink,
  ChevronRight, ArrowRight, ArrowDownLeft, ArrowUpRight,
  GitCommit, ShieldAlert, ShieldCheck, Zap, SlidersHorizontal,
  Layers, Database, Sparkles, AlertTriangle, ArrowUpDown,
  CornerDownRight, Filter
} from 'lucide-react';
import { fetchTracePattern } from '../utils/apiService';

// ─── Helper: derive pattern from features ─────────────────────────────────────
function detectPattern(feats) {
  const inDeg = feats.in_degree || 0;
  const outDeg = feats.out_degree || 0;
  if (inDeg === 0 && outDeg === 0) return '—';
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

// ─── Helper: severity badge styling ──────────────────────────────────────────
function severityStyle(severity) {
  const s = (severity || '').toUpperCase();
  if (s === 'HIGH' || s === 'CRITICAL') {
    return { background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' };
  }
  if (s === 'MEDIUM') {
    return { background: 'rgba(245,158,11,0.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.25)' };
  }
  return { background: 'rgba(16,185,129,0.1)', color: '#10b981', border: '1px solid rgba(16,185,129,0.25)' };
}

// ─── Helper: risk score badge styling ────────────────────────────────────────
function riskScoreStyle(score) {
  if (score >= 0.7) return { background: 'rgba(239,68,68,0.1)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.2)' };
  if (score >= 0.4) return { background: 'rgba(245,158,11,0.1)', color: '#f59e0b', border: '1px solid rgba(245,158,11,0.2)' };
  return { background: 'rgba(16,185,129,0.1)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' };
}

export default function Records() {
  const {
    globalJobId, setGlobalJobId, jobsList,
    cachedGraphElements, cachedAlerts,
    dataReady, dataLoading,
  } = useOutletContext();
  const navigate = useNavigate();

  // ── State ───────────────────────────────────────────────────
  const [selectedWallet, setSelectedWallet] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [severityFilter, setSeverityFilter] = useState('ALL');
  const [patternFilter, setPatternFilter] = useState('ALL');
  const [sortKey, setSortKey] = useState('RISK_DESC');
  const [drawerTab, setDrawerTab] = useState('HOPS');
  const [traceData, setTraceData] = useState(null);
  const [isTraceLoading, setIsTraceLoading] = useState(false);
  const [copiedId, setCopiedId] = useState(null);

  const completedJobs = (jobsList || []).filter(j => j.status === 'completed');
  const elements = cachedGraphElements || [];
  const alerts = cachedAlerts || [];

  // Nodes & Edges
  const nodes = useMemo(() => elements.filter(el => el.data && !el.data.source), [elements]);
  const edges = useMemo(() => elements.filter(el => el.data && el.data.source), [elements]);

  // Node features lookup
  const nodeFeats = useMemo(() => {
    const lookup = {};
    nodes.forEach(n => { lookup[n.data.id] = n.data.wallet_features || {}; });
    return lookup;
  }, [nodes]);

  // Enrich alerts with features and pattern
  const enrichedAlerts = useMemo(() => {
    return alerts.map(a => {
      const feats = nodeFeats[a.wallet_id] || {};
      const score = a.fusion_score || a.risk_probability || 0;
      return {
        ...a,
        score,
        pattern: detectPattern(feats),
        inDegree: feats.in_degree || 0,
        outDegree: feats.out_degree || 0,
        totalDegree: feats.degree || (feats.in_degree || 0) + (feats.out_degree || 0),
        totalSent: feats.total_sent_sats || 0,
        totalReceived: feats.total_received_sats || 0,
        pagerank: feats.pagerank || 0,
        betweenness: feats.betweenness_centrality || 0,
        clustering: feats.clustering_coefficient || 0,
        community: feats.community_id || '0',
      };
    });
  }, [alerts, nodeFeats]);

  // Filtered & Sorted Alerts
  const filteredAlerts = useMemo(() => {
    let result = enrichedAlerts.filter(item => {
      // Search
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase().trim();
        if (!item.wallet_id.toLowerCase().includes(q)) return false;
      }
      // Severity
      if (severityFilter !== 'ALL') {
        const sev = (item.severity || '').toUpperCase();
        if (severityFilter === 'HIGH_CRITICAL') {
          if (sev !== 'HIGH' && sev !== 'CRITICAL') return false;
        } else if (sev !== severityFilter) {
          return false;
        }
      }
      // Pattern
      if (patternFilter !== 'ALL') {
        if (patternFilter === 'FAN_OUT' && !item.pattern.includes('Fan-out')) return false;
        if (patternFilter === 'FAN_IN' && !item.pattern.includes('Fan-in')) return false;
        if (patternFilter === 'PEELING' && !item.pattern.includes('Pass-through')) return false;
      }
      return true;
    });

    // Sort
    result.sort((a, b) => {
      if (sortKey === 'RISK_DESC') return b.score - a.score;
      if (sortKey === 'RISK_ASC') return a.score - b.score;
      if (sortKey === 'DEGREE_DESC') return b.totalDegree - a.totalDegree;
      if (sortKey === 'ID_ASC') return a.wallet_id.localeCompare(b.wallet_id);
      return 0;
    });

    return result;
  }, [enrichedAlerts, searchQuery, severityFilter, patternFilter, sortKey]);

  // Fetch hop trace sequence when selected wallet changes
  useEffect(() => {
    if (!selectedWallet || !globalJobId) {
      setTraceData(null);
      return;
    }

    let active = true;
    setIsTraceLoading(true);
    setDrawerTab('HOPS');

    fetchTracePattern(globalJobId, selectedWallet.wallet_id)
      .then(res => {
        if (active) setTraceData(res);
      })
      .catch(err => {
        console.warn('Could not fetch hop trace for wallet:', err);
        if (active) setTraceData(null);
      })
      .finally(() => {
        if (active) setIsTraceLoading(false);
      });

    return () => { active = false; };
  }, [selectedWallet, globalJobId]);

  // Immediate connected edges from cache
  const connectedEdges = useMemo(() => {
    if (!selectedWallet) return { incoming: [], outgoing: [] };
    const wid = selectedWallet.wallet_id;
    const incoming = edges.filter(e => e.data.target === wid);
    const outgoing = edges.filter(e => e.data.source === wid);
    return { incoming, outgoing };
  }, [selectedWallet, edges]);

  const handleCopy = (id, e) => {
    if (e) e.stopPropagation();
    navigator.clipboard.writeText(id);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1800);
  };

  const handleInvestigateInGraph = (walletId) => {
    navigate('/graph', { state: { search: walletId } });
  };

  // Close drawer on Escape
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') setSelectedWallet(null);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // ── No data state ──────────────────────────────────────────
  if (!dataReady && !dataLoading && !globalJobId) {
    return (
      <div style={{ padding: '32px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: 'rgba(37,99,235,0.08)', padding: '24px', borderRadius: '50%', marginBottom: 20 }}>
          <Layers size={54} color="var(--accent)" />
        </div>
        <h2 style={{ fontWeight: 700, fontSize: '1.4rem', marginBottom: 8, color: 'var(--text-primary)' }}>
          No Transaction Records Available
        </h2>
        <p style={{ color: 'var(--text-muted)', maxWidth: 460, textAlign: 'center', lineHeight: 1.6, fontSize: '0.9rem' }}>
          Upload a Bitcoin transaction dataset on the <strong>Network Graph</strong> page to generate forensic audit records and topological alerts.
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
    <div style={{ display: 'flex', height: '100%', overflow: 'hidden', position: 'relative' }}>
      
      {/* ══ MAIN TABLE CONTAINER ════════════════════════════════════ */}
      <div style={{ flex: 1, padding: '28px 32px', overflowY: 'auto', minWidth: 0 }}>
        
        {/* Top Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '22px', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <h1 style={{ fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
                Wallet Records & Forensic Alerts
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
                  {globalJobId}
                </span>
              )}
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', marginTop: 4 }}>
              Click any record to inspect its multi-hop flow chain, graph centrality, and model explanations
            </p>
          </div>

          {/* Job Dropdown & Nav Buttons */}
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
                  <option value="">Select a Job</option>
                  {completedJobs.map(j => (
                    <option key={j.job_id} value={j.job_id}>
                      {j.job_id}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>
        </div>

        {/* Filters & Search Toolbar */}
        <div className="data-card" style={{ padding: '14px 18px', marginBottom: '18px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '14px', flexWrap: 'wrap' }}>
            
            {/* Search Input */}
            <div style={{
              display: 'flex', alignItems: 'center', gap: '8px',
              background: 'var(--bg-deep)', border: '1px solid var(--border)',
              borderRadius: '8px', padding: '8px 12px', flex: '1 1 240px', maxWidth: 360
            }}>
              <Search size={15} color="var(--text-muted)" />
              <input
                type="text"
                placeholder="Search by Wallet ID..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                style={{
                  background: 'transparent', border: 'none', outline: 'none',
                  fontSize: '0.84rem', color: 'var(--text-primary)', width: '100%'
                }}
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', display: 'flex' }}
                >
                  <X size={14} />
                </button>
              )}
            </div>

            {/* Severity Filter Chips */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginRight: 4 }}>
                Severity:
              </span>
              {[
                { id: 'ALL', label: 'All' },
                { id: 'HIGH_CRITICAL', label: 'High / Critical' },
                { id: 'MEDIUM', label: 'Medium' },
                { id: 'LOW', label: 'Low' },
              ].map(f => (
                <button
                  key={f.id}
                  onClick={() => setSeverityFilter(f.id)}
                  style={{
                    background: severityFilter === f.id ? 'var(--accent)' : 'var(--bg-deep)',
                    color: severityFilter === f.id ? '#ffffff' : 'var(--text-secondary)',
                    border: '1px solid ' + (severityFilter === f.id ? 'var(--accent)' : 'var(--border)'),
                    padding: '5px 11px', borderRadius: '6px', fontSize: '0.75rem',
                    fontWeight: 500, cursor: 'pointer', transition: 'all 0.12s'
                  }}
                >
                  {f.label}
                </button>
              ))}
            </div>

            {/* Sort Dropdown */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: 'auto' }}>
              <ArrowUpDown size={14} color="var(--text-muted)" />
              <select
                value={sortKey}
                onChange={(e) => setSortKey(e.target.value)}
                style={{
                  background: 'var(--bg-deep)', border: '1px solid var(--border)',
                  borderRadius: '6px', padding: '6px 10px', fontSize: '0.78rem',
                  color: 'var(--text-primary)', outline: 'none', cursor: 'pointer', fontWeight: 500
                }}
              >
                <option value="RISK_DESC">Sort: Highest Risk</option>
                <option value="RISK_ASC">Sort: Lowest Risk</option>
                <option value="DEGREE_DESC">Sort: Most Connections</option>
                <option value="ID_ASC">Sort: Wallet ID (A-Z)</option>
              </select>
            </div>

          </div>
        </div>

        {/* Results count banner */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', padding: '0 4px', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          <span>
            Showing <strong style={{ color: 'var(--text-primary)' }}>{filteredAlerts.length}</strong> of {enrichedAlerts.length} wallet records
          </span>
          {selectedWallet && (
            <span style={{ color: 'var(--accent)', fontWeight: 500 }}>
              Inspecting <strong>{selectedWallet.wallet_id}</strong> (Esc to close)
            </span>
          )}
        </div>

        {/* Records Table */}
        <div className="data-card" style={{ padding: 0, overflow: 'hidden', boxShadow: '0 2px 10px rgba(0,0,0,0.03)' }}>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.84rem' }}>
              <thead>
                <tr style={{ background: 'var(--bg-hover)', borderBottom: '1px solid var(--border)' }}>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Wallet ID</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Risk Score</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Severity</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Topology Pattern</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Flow (In / Out)</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Volume Handled</th>
                  <th style={{ padding: '13px 16px', fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.76rem', textTransform: 'uppercase', letterSpacing: '0.04em', textAlign: 'right' }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {filteredAlerts.length > 0 ? (
                  filteredAlerts.map((r) => {
                    const isSelected = selectedWallet?.wallet_id === r.wallet_id;
                    const totalVolume = (r.totalSent || 0) + (r.totalReceived || 0);

                    return (
                      <tr
                        key={r.wallet_id}
                        onClick={() => setSelectedWallet(isSelected ? null : r)}
                        style={{
                          borderBottom: '1px solid var(--border)',
                          cursor: 'pointer',
                          background: isSelected ? 'rgba(37,99,235,0.06)' : 'transparent',
                          transition: 'background 0.12s',
                        }}
                        onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.background = 'var(--bg-hover)'; }}
                        onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.background = 'transparent'; }}
                      >
                        {/* Wallet ID */}
                        <td style={{ padding: '13px 16px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span style={{
                              fontFamily: 'monospace', fontWeight: 600,
                              color: isSelected ? 'var(--accent)' : 'var(--text-primary)',
                              fontSize: '0.85rem'
                            }}>
                              {r.wallet_id}
                            </span>
                            <button
                              onClick={(e) => handleCopy(r.wallet_id, e)}
                              title="Copy Wallet ID"
                              style={{
                                background: 'none', border: 'none', cursor: 'pointer',
                                color: copiedId === r.wallet_id ? 'var(--success)' : 'var(--text-muted)',
                                padding: 2, display: 'flex', alignItems: 'center'
                              }}
                            >
                              {copiedId === r.wallet_id ? <Check size={12} /> : <Copy size={12} />}
                            </button>
                          </div>
                        </td>

                        {/* Risk Score */}
                        <td style={{ padding: '13px 16px' }}>
                          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '6px' }}>
                            <span style={{
                              ...riskScoreStyle(r.score),
                              padding: '3px 9px', borderRadius: '6px',
                              fontWeight: 700, fontSize: '0.78rem'
                            }}>
                              {(r.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        </td>

                        {/* Severity */}
                        <td style={{ padding: '13px 16px' }}>
                          <span style={{
                            ...severityStyle(r.severity),
                            padding: '3px 8px', borderRadius: '5px',
                            fontWeight: 600, fontSize: '0.72rem', textTransform: 'uppercase',
                            letterSpacing: '0.03em'
                          }}>
                            {(r.severity || '—').toUpperCase()}
                          </span>
                        </td>

                        {/* Pattern */}
                        <td style={{ padding: '13px 16px' }}>
                          <span style={{
                            color: r.pattern === '—' ? 'var(--text-muted)' : 'var(--text-primary)',
                            fontWeight: r.pattern === '—' ? 400 : 500,
                            fontSize: '0.82rem',
                            background: r.pattern !== '—' ? 'var(--bg-deep)' : 'transparent',
                            padding: r.pattern !== '—' ? '3px 8px' : 0,
                            borderRadius: '4px',
                            border: r.pattern !== '—' ? '1px solid var(--border)' : 'none'
                          }}>
                            {r.pattern}
                          </span>
                        </td>

                        {/* In / Out Flow */}
                        <td style={{ padding: '13px 16px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.76rem', color: 'var(--text-secondary)' }}>
                            <span title="Incoming connections (senders)">
                              <ArrowDownLeft size={12} color="#10b981" style={{ verticalAlign: 'middle', marginRight: 2 }} />
                              {r.inDegree} in
                            </span>
                            <span style={{ color: 'var(--border)' }}>•</span>
                            <span title="Outgoing connections (receivers)">
                              <ArrowUpRight size={12} color="#ef4444" style={{ verticalAlign: 'middle', marginRight: 2 }} />
                              {r.outDegree} out
                            </span>
                          </div>
                        </td>

                        {/* Volume */}
                        <td style={{ padding: '13px 16px', color: 'var(--text-primary)', fontWeight: 500, fontSize: '0.8rem' }}>
                          {formatSatoshis(totalVolume)}
                        </td>

                        {/* Action */}
                        <td style={{ padding: '13px 16px', textAlign: 'right' }}>
                          <button
                            style={{
                              background: isSelected ? 'var(--accent)' : 'transparent',
                              color: isSelected ? '#ffffff' : 'var(--text-muted)',
                              border: isSelected ? 'none' : '1px solid var(--border)',
                              borderRadius: '6px',
                              padding: '5px 10px',
                              fontSize: '0.74rem',
                              fontWeight: 500,
                              cursor: 'pointer',
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '4px',
                              transition: 'all 0.12s'
                            }}
                          >
                            <span>Inspect</span>
                            <ChevronRight size={12} />
                          </button>
                        </td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={7} style={{ padding: '36px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No records match the current filter or search criteria.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </div>

      {/* ══ SLIDE-OVER FORENSIC INSPECTION DRAWER ════════════════════ */}
      {selectedWallet && (
        <div style={{
          width: 480,
          background: 'var(--bg-panel)',
          borderLeft: '1px solid var(--border)',
          display: 'flex',
          flexDirection: 'column',
          height: '100%',
          boxShadow: '-4px 0 20px rgba(0,0,0,0.06)',
          zIndex: 30,
          flexShrink: 0
        }}>
          
          {/* Drawer Header */}
          <div style={{
            padding: '20px 22px', borderBottom: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start'
          }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  fontFamily: 'monospace', fontSize: '1.05rem', fontWeight: 700,
                  color: 'var(--text-primary)'
                }}>
                  {selectedWallet.wallet_id}
                </span>
                <button
                  onClick={() => handleCopy(selectedWallet.wallet_id)}
                  title="Copy Wallet ID"
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: copiedId === selectedWallet.wallet_id ? 'var(--success)' : 'var(--text-muted)',
                    padding: 2, display: 'flex'
                  }}
                >
                  {copiedId === selectedWallet.wallet_id ? <Check size={14} /> : <Copy size={14} />}
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: 8 }}>
                <span style={{
                  ...riskScoreStyle(selectedWallet.score),
                  padding: '3px 8px', borderRadius: '5px', fontWeight: 700, fontSize: '0.76rem'
                }}>
                  Risk {(selectedWallet.score * 100).toFixed(1)}%
                </span>
                <span style={{
                  ...severityStyle(selectedWallet.severity),
                  padding: '3px 8px', borderRadius: '5px', fontWeight: 600, fontSize: '0.72rem',
                  textTransform: 'uppercase'
                }}>
                  {selectedWallet.severity || 'HIGH'}
                </span>
                <span style={{
                  background: 'var(--bg-deep)', padding: '3px 8px', borderRadius: '5px',
                  fontSize: '0.74rem', color: 'var(--text-secondary)', border: '1px solid var(--border)'
                }}>
                  {selectedWallet.pattern}
                </span>
              </div>
            </div>

            <button
              onClick={() => setSelectedWallet(null)}
              style={{
                background: 'var(--bg-deep)', border: '1px solid var(--border)',
                borderRadius: '6px', padding: '6px', cursor: 'pointer',
                color: 'var(--text-secondary)', display: 'flex', alignItems: 'center'
              }}
              title="Close drawer (Esc)"
            >
              <X size={15} />
            </button>
          </div>

          {/* Drawer Tabs */}
          <div style={{
            display: 'flex', borderBottom: '1px solid var(--border)',
            background: 'var(--bg-card)', padding: '0 16px'
          }}>
            {[
              { id: 'HOPS', label: 'Hop Flow Chain', icon: GitCommit },
              { id: 'FEATURES', label: 'Graph Metrics', icon: Network },
              { id: 'MODELS', label: 'AI & SHAP', icon: ShieldAlert },
            ].map(tab => {
              const Icon = tab.icon;
              const isActive = drawerTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setDrawerTab(tab.id)}
                  style={{
                    display: 'flex', alignItems: 'center', gap: '6px',
                    padding: '12px 14px', border: 'none', background: 'transparent',
                    borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
                    color: isActive ? 'var(--accent)' : 'var(--text-muted)',
                    fontWeight: isActive ? 600 : 500, fontSize: '0.8rem',
                    cursor: 'pointer', transition: 'all 0.12s'
                  }}
                >
                  <Icon size={14} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>

          {/* Drawer Body (Scrollable) */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '20px 22px' }}>
            
            {/* ── TAB 1: HOP FLOW CHAIN ───────────────────────────── */}
            {drawerTab === 'HOPS' && (
              <div>
                {/* Summary Banner */}
                <div style={{
                  background: 'var(--bg-deep)', border: '1px solid var(--border)',
                  borderRadius: '10px', padding: '12px 16px', marginBottom: '18px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                        Flow Sequence Path
                      </span>
                      <div style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>
                        {traceData ? `${traceData.hop_count || 0} Hops Traced` : 'Direct Network Hops'}
                      </div>
                    </div>
                    {traceData?.total_amount_sats !== undefined && (
                      <div style={{ textAlign: 'right' }}>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                          Path Volume
                        </span>
                        <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--accent)', marginTop: 2 }}>
                          {formatSatoshis(traceData.total_amount_sats)}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {/* Step Timeline */}
                {isTraceLoading ? (
                  <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.84rem' }}>
                    Tracing multi-hop fund flow…
                  </div>
                ) : traceData?.sequence && traceData.sequence.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', position: 'relative' }}>
                    {traceData.sequence.map((step, idx) => {
                      const isSource = (step.source_wallet || step.source) === selectedWallet.wallet_id;
                      const isTarget = (step.target_wallet || step.target) === selectedWallet.wallet_id;

                      return (
                        <div
                          key={idx}
                          style={{
                            display: 'flex', gap: '12px',
                            background: 'var(--bg-card)', border: '1px solid var(--border)',
                            borderRadius: '10px', padding: '12px 14px',
                            boxShadow: '0 1px 4px rgba(0,0,0,0.02)'
                          }}
                        >
                          <div style={{
                            width: 26, height: 26, borderRadius: '50%',
                            background: 'rgba(37,99,235,0.1)', color: 'var(--accent)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '0.75rem', fontWeight: 700, flexShrink: 0
                          }}>
                            {step.step || idx + 1}
                          </div>

                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.8rem', flexWrap: 'wrap' }}>
                              <span style={{
                                fontFamily: 'monospace', fontWeight: isSource ? 700 : 500,
                                color: isSource ? 'var(--accent)' : 'var(--text-primary)'
                              }}>
                                {step.source_wallet || step.source}
                              </span>
                              <ArrowRight size={13} color="var(--text-muted)" />
                              <span style={{
                                fontFamily: 'monospace', fontWeight: isTarget ? 700 : 500,
                                color: isTarget ? 'var(--accent)' : 'var(--text-primary)'
                              }}>
                                {step.target_wallet || step.target}
                              </span>
                            </div>

                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 6, fontSize: '0.74rem' }}>
                              <span style={{ color: 'var(--danger)', fontWeight: 600 }}>
                                {formatSatoshis(step.amount_sats)}
                              </span>
                              {step.txid && (
                                <span style={{ color: 'var(--text-muted)', fontFamily: 'monospace' }}>
                                  Tx: {step.txid.substring(0, 10)}…
                                </span>
                              )}
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  /* Fallback to direct connected neighbors from graph cache */
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                    <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                      Multi-hop path sequence empty. Showing direct 1-hop connected neighbors from graph:
                    </div>

                    {/* Incoming Peers */}
                    <div>
                      <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                        Incoming Senders ({connectedEdges.incoming.length})
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {connectedEdges.incoming.slice(0, 5).map((e, idx) => (
                          <div key={idx} style={{
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            background: 'var(--bg-deep)', padding: '7px 10px', borderRadius: '6px',
                            fontSize: '0.76rem', border: '1px solid var(--border)'
                          }}>
                            <span style={{ fontFamily: 'monospace', color: 'var(--text-primary)' }}>
                              {e.data.source}
                            </span>
                            <span style={{ color: '#10b981', fontWeight: 600 }}>
                              +{formatSatoshis(e.data.total_amount_sats || e.data.weight)}
                            </span>
                          </div>
                        ))}
                        {connectedEdges.incoming.length === 0 && (
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>No incoming senders recorded</div>
                        )}
                      </div>
                    </div>

                    {/* Outgoing Peers */}
                    <div>
                      <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 8 }}>
                        Outgoing Recipients ({connectedEdges.outgoing.length})
                      </div>
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                        {connectedEdges.outgoing.slice(0, 5).map((e, idx) => (
                          <div key={idx} style={{
                            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                            background: 'var(--bg-deep)', padding: '7px 10px', borderRadius: '6px',
                            fontSize: '0.76rem', border: '1px solid var(--border)'
                          }}>
                            <span style={{ fontFamily: 'monospace', color: 'var(--text-primary)' }}>
                              {e.data.target}
                            </span>
                            <span style={{ color: '#ef4444', fontWeight: 600 }}>
                              -{formatSatoshis(e.data.total_amount_sats || e.data.weight)}
                            </span>
                          </div>
                        ))}
                        {connectedEdges.outgoing.length === 0 && (
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>No outgoing recipients recorded</div>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── TAB 2: GRAPH METRICS & TOPOLOGY ─────────────────── */}
            {drawerTab === 'FEATURES' && (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  {[
                    { label: 'Total Degree', val: selectedWallet.totalDegree, sub: 'Total Links' },
                    { label: 'In / Out Degree', val: `${selectedWallet.inDegree} in / ${selectedWallet.outDegree} out`, sub: 'Degree Split' },
                    { label: 'Total Sent', val: formatSatoshis(selectedWallet.totalSent), sub: 'Outflow' },
                    { label: 'Total Received', val: formatSatoshis(selectedWallet.totalReceived), sub: 'Inflow' },
                    { label: 'PageRank', val: (selectedWallet.pagerank || 0).toFixed(6), sub: 'Centrality Authority' },
                    { label: 'Betweenness', val: (selectedWallet.betweenness || 0).toFixed(6), sub: 'Bridge Metric' },
                    { label: 'Clustering Coeff.', val: (selectedWallet.clustering || 0).toFixed(4), sub: 'Local Density' },
                    { label: 'Community Cluster', val: `ID: ${selectedWallet.community}`, sub: 'Subgraph Group' },
                  ].map(item => (
                    <div
                      key={item.label}
                      style={{
                        background: 'var(--bg-deep)', border: '1px solid var(--border)',
                        borderRadius: '8px', padding: '12px 14px'
                      }}
                    >
                      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 600 }}>
                        {item.label}
                      </div>
                      <div style={{ fontSize: '0.94rem', fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>
                        {item.val}
                      </div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: 2 }}>
                        {item.sub}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* ── TAB 3: AI MODELS & SHAP EXPLAINABILITY ──────────── */}
            {drawerTab === 'MODELS' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '18px' }}>
                
                {/* Model Scorecards */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10 }}>
                    Independent AI Model Evaluations
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {[
                      {
                        name: 'GraphSAGE GNN',
                        score: selectedWallet.risk_probability || selectedWallet.score,
                        flagged: selectedWallet.risk_prediction === 1,
                        desc: 'Structural neighborhood embedding'
                      },
                      {
                        name: 'Isolation Forest',
                        score: selectedWallet.isolation_forest_score || 0,
                        flagged: (selectedWallet.isolation_forest_score || 0) > 0.5,
                        desc: 'High-dimensional feature outlier'
                      },
                      {
                        name: 'Autoencoder',
                        score: selectedWallet.autoencoder_score || 0,
                        flagged: (selectedWallet.autoencoder_score || 0) > 0.5,
                        desc: 'Reconstruction error anomaly'
                      },
                      {
                        name: 'Deterministic Rules',
                        score: selectedWallet.deterministic_score || 0,
                        flagged: (selectedWallet.deterministic_score || 0) > 0.5,
                        desc: 'Dispersal & threshold heuristics'
                      },
                    ].map(m => (
                      <div
                        key={m.name}
                        style={{
                          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                          background: 'var(--bg-deep)', border: '1px solid var(--border)',
                          borderRadius: '8px', padding: '10px 14px'
                        }}
                      >
                        <div>
                          <div style={{ fontWeight: 600, fontSize: '0.8rem', color: 'var(--text-primary)' }}>
                            {m.name}
                          </div>
                          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                            {m.desc}
                          </div>
                        </div>

                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <span style={{ fontSize: '0.8rem', fontWeight: 700, color: m.flagged ? '#ef4444' : '#10b981' }}>
                            {(m.score * 100).toFixed(1)}%
                          </span>
                          <span style={{
                            padding: '2px 7px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 600,
                            background: m.flagged ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                            color: m.flagged ? '#ef4444' : '#10b981'
                          }}>
                            {m.flagged ? 'FLAGGED' : 'CLEAN'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* SHAP Factor Breakdown */}
                <div>
                  <div style={{ fontSize: '0.74rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 10 }}>
                    SHAP Risk Contribution Factors
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {selectedWallet.top_shap_factors && selectedWallet.top_shap_factors.length > 0 ? (
                      selectedWallet.top_shap_factors.map((f, i) => {
                        const feat = (f.feature || 'unknown').replace(/_/g, ' ');
                        const val = f.shap_value || f.contribution || 0;
                        const isRisk = val >= 0;

                        return (
                          <div
                            key={i}
                            style={{
                              background: 'var(--bg-card)', border: '1px solid var(--border)',
                              borderRadius: '8px', padding: '9px 12px'
                            }}
                          >
                            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{feat}</span>
                              <span style={{ fontWeight: 600, color: isRisk ? '#ef4444' : '#10b981' }}>
                                {isRisk ? '+' : ''}{val.toFixed(3)}
                              </span>
                            </div>
                            <div style={{ height: 4, background: 'var(--bg-deep)', borderRadius: 2, marginTop: 6, overflow: 'hidden' }}>
                              <div style={{
                                height: '100%',
                                width: `${Math.min(100, Math.abs(val) * 100)}%`,
                                background: isRisk ? '#ef4444' : '#10b981'
                              }} />
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        No specific SHAP factors recorded for this entity.
                      </div>
                    )}
                  </div>
                </div>

              </div>
            )}

          </div>

          {/* Drawer Footer Actions */}
          <div style={{
            padding: '16px 22px', borderTop: '1px solid var(--border)',
            display: 'flex', gap: '10px', background: 'var(--bg-card)'
          }}>
            <button
              className="action-btn primary"
              style={{ flex: 1, padding: '10px 14px', borderRadius: '8px', fontSize: '0.82rem' }}
              onClick={() => handleInvestigateInGraph(selectedWallet.wallet_id)}
            >
              <Network size={14} /> Open in Network Graph
            </button>
            <button
              className="action-btn outline"
              style={{ width: 'auto', padding: '10px 14px', borderRadius: '8px', fontSize: '0.82rem' }}
              onClick={() => handleCopy(selectedWallet.wallet_id)}
            >
              <Copy size={14} /> Copy ID
            </button>
          </div>

        </div>
      )}

    </div>
  );
}
