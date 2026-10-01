import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useOutletContext, useLocation } from 'react-router-dom';
import {
  Search, Plus, Database, History, UploadCloud,
  Loader, AlertTriangle, CheckCircle, X, ChevronRight,
  Network, RefreshCw, Zap
} from 'lucide-react';
import GraphView from '../components/GraphView';
import Sidebar from '../components/Sidebar';
import {
  uploadCSV, pollJobStatus, fetchGraphData,
  fetchAlerts
} from '../utils/apiService';

// ─── Onboarding Tour Steps ───────────────────────────────────────────────────
const TOUR_STEPS = [
  {
    icon: '🔍',
    iconBg: 'rgba(79,142,247,0.15)',
    title: 'Bitcoin Forensic Analyst',
    desc: 'This tool detects money laundering, smurfing, and shell networks in Bitcoin transactions using Graph Neural Networks, ML models, and on-device AI explanations. Let\'s walk through how it works.',
  },
  {
    icon: '📂',
    iconBg: 'rgba(61,214,140,0.12)',
    title: 'Upload Your Transaction Data',
    desc: 'Click "New Analysis" in the left sidebar and upload all 6 required CSV files: wallets, transactions, inputs, outputs, network_observations, and ip_metadata. The forensic pipeline runs automatically.',
  },
  {
    icon: '🕸️',
    iconBg: 'rgba(155,115,245,0.12)',
    title: 'Explore the Transaction Graph',
    desc: 'Each node is a wallet. Red nodes are flagged anomalies. Green nodes are normal. Click any node to see its full risk profile — GNN embeddings, ML risk scores, SHAP explanations, and more.',
  },
  {
    icon: '🤖',
    iconBg: 'rgba(240,161,66,0.12)',
    title: 'AI-Powered Explanations',
    desc: 'Switch to Phase 4: LLM in the right panel to get a natural language forensic summary generated entirely on your device using a local Qwen 2.5 model — no data sent to external servers.',
  },
  {
    icon: '⚡',
    iconBg: 'rgba(79,142,247,0.15)',
    title: 'Trace Fund Flows',
    desc: 'Click "Dynamic Trace Flow" to animate how money moves through the network. The system detects Cycle, Fan-Out, Fan-In, and Peeling Chain patterns in real time.',
  },
];

// ─── Required CSV files ──────────────────────────────────────────────────────
const REQUIRED_KEYS = [
  'wallets', 'transactions', 'transaction_inputs',
  'transaction_outputs', 'network_observations', 'ip_metadata'
];

// ─── Tour Modal ──────────────────────────────────────────────────────────────
function TourModal({ onClose }) {
  const [step, setStep] = useState(0);
  const current = TOUR_STEPS[step];
  const isLast = step === TOUR_STEPS.length - 1;

  return (
    <div className="tour-backdrop" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="tour-modal">
        <div className="tour-step-indicator">
          {TOUR_STEPS.map((_, i) => (
            <div
              key={i}
              className={`tour-step-dot ${i < step ? 'done' : i === step ? 'active' : ''}`}
            />
          ))}
        </div>

        <div className="tour-icon" style={{ background: current.iconBg }}>
          {current.icon}
        </div>

        <h2>{current.title}</h2>
        <p>{current.desc}</p>

        <div className="tour-modal-actions">
          <button className="tour-skip" onClick={onClose}>Skip tour</button>
          <div className="tour-nav">
            {step > 0 && (
              <button
                className="action-btn outline"
                style={{ width: 'auto', padding: '9px 20px' }}
                onClick={() => setStep(s => s - 1)}
              >
                Back
              </button>
            )}
            <button
              className="action-btn primary"
              style={{ width: 'auto', padding: '9px 22px' }}
              onClick={() => isLast ? onClose() : setStep(s => s + 1)}
            >
              {isLast ? 'Get Started' : 'Next'} {!isLast && <ChevronRight size={15} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Upload Modal ────────────────────────────────────────────────────────────
function UploadModal({ onClose, onUpload }) {
  const [matchedFiles, setMatchedFiles] = useState({});
  const fileRef = useRef();

  const handleFiles = (files) => {
    const map = {};
    Array.from(files).forEach(f => {
      const name = f.name.toLowerCase().replace(/[^a-z0-9_]/g, '_');
      REQUIRED_KEYS.forEach(k => { if (name.includes(k)) map[k] = f; });
    });
    setMatchedFiles(map);
  };

  const canUpload = REQUIRED_KEYS.every(k => matchedFiles[k]);

  return (
    <div className="upload-modal" onClick={(e) => e.target === e.currentTarget && onClose()}>
      <div className="upload-modal-box">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h2>Upload Transaction Data</h2>
          <button className="error-banner-close" onClick={onClose}><X size={18} /></button>
        </div>
        <p>Select all 6 required CSV files at once. You can select multiple files in one picker.</p>

        <ul className="file-list">
          {REQUIRED_KEYS.map(k => (
            <li key={k} className={matchedFiles[k] ? 'matched' : 'missing'}>
              {matchedFiles[k] ? <CheckCircle size={13} /> : <div style={{ width: 13, height: 13, borderRadius: '50%', border: '1px solid currentColor', opacity: 0.4 }} />}
              <span style={{ fontFamily: 'monospace' }}>{k}.csv</span>
              {matchedFiles[k] && <span style={{ marginLeft: 'auto', opacity: 0.6 }}>{(matchedFiles[k].size / 1024).toFixed(0)} KB</span>}
            </li>
          ))}
        </ul>

        <input ref={fileRef} type="file" accept=".csv" multiple style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)} />

        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <button className="action-btn outline" onClick={() => fileRef.current.click()}>
            <UploadCloud size={15} /> Choose Files
          </button>
          <button
            className="action-btn primary"
            disabled={!canUpload}
            onClick={() => { onClose(); onUpload(matchedFiles); }}
          >
            <Zap size={15} /> Run Forensic Pipeline
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Main Component ──────────────────────────────────────────────────────────
export default function NetworkGraph() {
  const {
    globalJobId, setGlobalJobId,
    jobsList, refreshJobs,
    cachedGraphElements, setCachedGraphElements,
    cachedAlerts, setCachedAlerts,
    dataLoading, dataReady,
    isDarkMode,
  } = useOutletContext();

  const [selectedNode, setSelectedNode] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadError, setUploadError] = useState(null);
  const [searchInput, setSearchInput] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  const [showTour, setShowTour] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);
  const location = useLocation();

  useEffect(() => {
    if (location.state?.search) {
      setSearchInput(location.state.search);
      setActiveSearch(location.state.search);
    }
  }, [location.state]);

  const graphRef = useRef();
  const pollIntervalRef = useRef(null);

  const stopPolling = () => {
    if (pollIntervalRef.current) { clearInterval(pollIntervalRef.current); pollIntervalRef.current = null; }
  };

  useEffect(() => () => stopPolling(), []);

  // Show tour for first-time visitors with no data
  useEffect(() => {
    if (!globalJobId && !dataReady && !localStorage.getItem('toured')) {
      setShowTour(true);
    }
  }, [globalJobId, dataReady]);

  const handleJobSelect = (jid) => {
    setGlobalJobId(jid || null);
  };

  const handleUpload = async (fileMap) => {
    setUploadError(null);
    stopPolling();
    try {
      setIsUploading(true);
      setUploadProgress(5);
      setSelectedNode(null);
      setStatusMessage('Uploading 6 CSV files...');

      let uploadRes;
      try { uploadRes = await uploadCSV(fileMap); }
      catch (err) {
        throw new Error(
          err.message.includes('fetch')
            ? 'Cannot connect to backend (port 8000). Is the server running?'
            : `Upload failed: ${err.message}`
        );
      }

      const cjid = uploadRes.job_id;
      setStatusMessage(`Job ${cjid} — pipeline starting...`);
      setUploadProgress(10);

      let errCount = 0;
      pollIntervalRef.current = setInterval(async () => {
        try {
          const st = await pollJobStatus(cjid);
          errCount = 0;
          setUploadProgress(st.progress || 0);
          setStatusMessage(`${st.current_stage || 'Processing'}… (${st.progress || 0}%)`);
          if (st.status === 'completed') {
            stopPolling();
            setUploadProgress(100);
            setStatusMessage('Pipeline complete — loading graph...');
            await refreshJobs();
            setGlobalJobId(cjid);   // This triggers Layout to fetch & cache
            setIsUploading(false);
          } else if (st.status === 'failed') {
            stopPolling();
            throw new Error(`Pipeline failed: ${st.error_message || 'See backend logs.'}`);
          }
        } catch (err) {
          if (++errCount >= 5) {
            stopPolling();
            setIsUploading(false);
            setUploadError(err.message);
          }
        }
      }, 1500);
    } catch (err) {
      stopPolling();
      setIsUploading(false);
      setUploadError(err.message);
    }
  };

  const handleSampleDataset = async () => {
    // Load the default pipeline output if available
    try {
      const els = await fetchGraphData(null);
      setCachedGraphElements(els);
      setCachedAlerts([]);
      setGlobalJobId(null);
    } catch (_) {
      setUploadError('No sample data found. Upload your own CSV files to begin.');
    }
  };

  // Derive counts from cached data
  const elements = cachedGraphElements;
  const anomalyCount = elements.filter(el => el.data && el.data.is_anomaly === true).length;
  const normalCount = elements.filter(el => el.data && !el.data.source && !el.data.is_anomaly).length;
  const completedJobs = jobsList.filter(j => j.status === 'completed');

  const showEmptyState = elements.length === 0 && !isUploading && !dataLoading;

  return (
    <div style={{ display: 'flex', height: '100%', width: '100%' }}>
      {/* ── Tour ── */}
      {showTour && (
        <TourModal onClose={() => { setShowTour(false); localStorage.setItem('toured', '1'); }} />
      )}

      {/* ── Upload Modal ── */}
      {showUploadModal && (
        <UploadModal onClose={() => setShowUploadModal(false)} onUpload={handleUpload} />
      )}

      {/* ══ CENTER CANVAS ════════════════════════════════════ */}
      <div className="canvas-area">
        {/* Top bar */}
        <div className="canvas-topbar">
          <form onSubmit={e => { e.preventDefault(); setActiveSearch(searchInput.trim()); }}
            className="search-box">
            <Search size={15} color="var(--text-muted)" />
            <input
              type="text"
              placeholder="Search wallet ID…"
              value={searchInput}
              onChange={e => setSearchInput(e.target.value)}
            />
          </form>

          {/* Action buttons */}
          <button className="action-btn primary" style={{ width: 'auto', padding: '9px 14px', borderRadius: '10px' }} onClick={() => setShowUploadModal(true)} disabled={isUploading}>
            <UploadCloud size={15} /> New Analysis
          </button>
          
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'var(--bg-panel)', border: '1px solid var(--border)', borderRadius: '10px', padding: '6px 10px' }}>
            <History size={14} color="var(--text-muted)" />
            <select
               value={globalJobId || ''}
               onChange={(e) => handleJobSelect(e.target.value)}
               style={{ background: 'transparent', color: 'var(--text-primary)', border: 'none', fontSize: '0.8rem', outline: 'none', cursor: 'pointer' }}
            >
               <option value="">Sample Dataset</option>
               {completedJobs.map(j => (
                  <option key={j.job_id} value={j.job_id} style={{ background: 'var(--bg-panel)' }}>
                    {j.job_id}
                  </option>
               ))}
            </select>
          </div>

          {anomalyCount > 0 && !isUploading && (
            <div className="stat-chip danger">
              <AlertTriangle size={13} />
              <span><strong>{anomalyCount}</strong> flagged</span>
            </div>
          )}
          {normalCount > 0 && !isUploading && (
            <div className="stat-chip success">
              <CheckCircle size={13} />
              <span><strong>{normalCount}</strong> normal</span>
            </div>
          )}
        </div>

        {/* Error banner */}
        {uploadError && !isUploading && (
          <div className="error-banner" style={{ top: 70, left: 16, transform: 'none', right: 'auto', width: 'auto', maxWidth: 480 }}>
            <AlertTriangle size={18} color="var(--danger)" style={{ flexShrink: 0, marginTop: 1 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 600, color: 'var(--danger)', fontSize: '0.83rem', marginBottom: 3 }}>Error</div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', whiteSpace: 'pre-wrap' }}>{uploadError}</div>
            </div>
            <button className="error-banner-close" onClick={() => setUploadError(null)}><X size={15} /></button>
          </div>
        )}

        {/* Upload progress */}
        {isUploading && (
          <div className="upload-progress">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
              <Loader size={14} className="spinner" color="var(--accent)" />
              <span style={{ fontSize: '0.83rem', fontWeight: 600 }}>Pipeline Running</span>
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 8 }}>{statusMessage}</div>
            <div className="progress-bar-track">
              <div className="progress-bar-fill" style={{ width: `${uploadProgress}%` }} />
            </div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 6, textAlign: 'right' }}>
              {uploadProgress}%
            </div>
          </div>
        )}

        {/* Main content */}
        {dataLoading ? (
          <div className="loading-overlay">
            <Loader size={40} className="spinner" color="var(--accent)" />
            <h3>Loading network visualization…</h3>
          </div>
        ) : showEmptyState ? (
          <div className="empty-state">
            <Network size={52} style={{ opacity: 0.2, marginBottom: 8 }} />
            <h2>No data loaded</h2>
            <p>
              Start a new analysis by uploading your 6 Bitcoin CSV files, or
              load the sample dataset to explore the interface.
            </p>
            <div style={{ display: 'flex', gap: 10, marginTop: 8 }}>
              <button className="action-btn primary" style={{ width: 'auto', padding: '10px 22px' }}
                onClick={() => setShowUploadModal(true)}>
                <Plus size={15} /> New Analysis
              </button>
              <button className="action-btn outline" style={{ width: 'auto', padding: '10px 22px' }}
                onClick={handleSampleDataset}>
                <Database size={15} /> Sample Dataset
              </button>
            </div>
            <button style={{ marginTop: 14, background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '0.78rem', cursor: 'pointer' }}
              onClick={() => setShowTour(true)}>
              <Zap size={12} style={{ marginRight: 4, verticalAlign: 'middle' }} />
              See how it works
            </button>
          </div>
        ) : (
          <GraphView
            ref={graphRef}
            elements={elements}
            onNodeSelect={setSelectedNode}
            searchQuery={activeSearch}
            jobId={globalJobId}
            isDarkMode={isDarkMode}
          />
        )}

        {/* Legend */}
        {elements.length > 0 && !dataLoading && !isUploading && (
          <div className="canvas-legend">
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#e85454' }} />
              Anomalous wallet
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#3dd68c' }} />
              Normal wallet
            </div>
            <div className="legend-item">
              <div className="legend-dot" style={{ background: '#fb923c', border: '2px solid #fb923c' }} />
              Cycle edges
            </div>
          </div>
        )}
      </div>

      {/* ══ RIGHT ENTITY PANEL ══════════════════════════════ */}
      <Sidebar selectedNode={selectedNode} onTrace={async (nodeId) => {
        if (graphRef.current) await graphRef.current.traceFunds(nodeId);
      }} jobId={globalJobId} />
    </div>
  );
}
