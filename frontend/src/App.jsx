import React, { useState, useEffect, useRef } from 'react';
import { Search, UploadCloud, Loader, AlertTriangle, CheckCircle2, History, XCircle, RefreshCw } from 'lucide-react';
import GraphView from './components/GraphView';
import Sidebar from './components/Sidebar';
import { uploadCSV, pollJobStatus, fetchGraphData, fetchJobsList, fetchAlerts } from './utils/apiService';

export default function App() {
  const [elements, setElements] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Jobs State
  const [jobsList, setJobsList] = useState([]);
  const [jobId, setJobId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [uploadError, setUploadError] = useState(null); // null | string
  const [alertsCount, setAlertsCount] = useState(0);
  
  const [searchInput, setSearchInput] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  
  const graphRef = useRef();
  const fileInputRef = useRef();
  const pollIntervalRef = useRef(null);

  // Cleanup polling on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
  }, []);

  // Load existing jobs on mount
  useEffect(() => {
    async function initJobs() {
      try {
        setLoading(true);
        const jobs = await fetchJobsList();
        setJobsList(jobs);

        const completedJobs = jobs.filter(j => j.status === 'completed');
        let initialJobId = null;

        if (completedJobs.length > 0) {
          initialJobId = completedJobs[0].job_id;
        }

        if (initialJobId) {
          setJobId(initialJobId);
          await loadGraphForJob(initialJobId);
        } else {
          // Fallback to default graph overview (from outputs/ directory)
          try {
            const defaultElements = await fetchGraphData(null);
            setElements(defaultElements);
          } catch (_) {
            // No default data either — show upload prompt
          }
        }
      } catch (err) {
        console.warn('Initial jobs loading error:', err);
      } finally {
        setLoading(false);
      }
    }

    initJobs();
  }, []);

  const loadGraphForJob = async (selectedJobId) => {
    try {
      setLoading(true);
      setSelectedNode(null);
      const graphElements = await fetchGraphData(selectedJobId);
      setElements(graphElements);
      
      try {
        const alertsData = await fetchAlerts(selectedJobId);
        setAlertsCount((alertsData.alerts || []).length);
      } catch (_) {
        setAlertsCount(0);
      }
    } catch (err) {
      console.error(`Failed to load graph for ${selectedJobId}:`, err);
    } finally {
      setLoading(false);
    }
  };

  const handleJobChange = async (e) => {
    const newJobId = e.target.value;
    setJobId(newJobId);
    await loadGraphForJob(newJobId);
  };

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
  };

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    
    // Reset file input so same files can be re-uploaded if needed
    e.target.value = '';
    
    const fileMap = {};
    const requiredKeys = ['wallets', 'transactions', 'transaction_inputs', 'transaction_outputs', 'network_observations', 'ip_metadata'];
    
    files.forEach(f => {
      const name = f.name.toLowerCase().replace(/[^a-z0-9_]/g, '_');
      requiredKeys.forEach(k => {
        if (name.includes(k)) fileMap[k] = f;
      });
    });
    
    const missing = requiredKeys.filter(k => !fileMap[k]);
    if (missing.length > 0) {
      setUploadError(`Please select all 6 required CSV files.\nMissing: ${missing.join(', ')}`);
      return;
    }
    
    // Clear any previous error
    setUploadError(null);
    stopPolling();
    
    try {
      setIsUploading(true);
      setUploadProgress(5);
      setElements([]);
      setSelectedNode(null);
      setStatusMessage('Uploading 6 CSV files to forensic backend...');
      
      // 1. Upload to backend API
      let uploadRes;
      try {
        uploadRes = await uploadCSV(fileMap);
      } catch (err) {
        const isConnRefused = err.message.includes('Failed to fetch') || err.message.includes('NetworkError') || err.message.includes('fetch');
        throw new Error(isConnRefused
          ? 'Cannot connect to backend. Make sure the server is running on port 8000.\n\nRun: cd backend && python main.py'
          : `Upload failed: ${err.message}`
        );
      }

      const currentJobId = uploadRes.job_id;
      setJobId(currentJobId);
      setStatusMessage(`Job ${currentJobId} created — pipeline starting...`);
      setUploadProgress(10);
      
      // 2. Poll Status API
      let consecutiveErrors = 0;
      pollIntervalRef.current = setInterval(async () => {
        try {
          const statusRes = await pollJobStatus(currentJobId);
          consecutiveErrors = 0;
          
          const progress = statusRes.progress || 0;
          const stage = statusRes.step || statusRes.current_stage || '';
          setUploadProgress(progress);
          setStatusMessage(`Pipeline running: ${stage} (${progress}%)`);
          
          if (statusRes.status === 'completed' || progress >= 100) {
            stopPolling();
            setUploadProgress(100);
            setStatusMessage('Pipeline complete — loading graph visualization...');
            
            const updatedJobs = await fetchJobsList();
            setJobsList(updatedJobs);

            await loadGraphForJob(currentJobId);
            setIsUploading(false);

          } else if (statusRes.status === 'failed') {
            stopPolling();
            throw new Error(`Pipeline failed: ${statusRes.error_message || 'Check backend logs for details.'}`);
          }
        } catch (err) {
          consecutiveErrors++;
          if (consecutiveErrors >= 5) {
            // 5 consecutive errors (7.5s) = backend likely crashed
            stopPolling();
            const isConnErr = err.message.includes('Failed to fetch') || err.message.includes('NetworkError');
            setIsUploading(false);
            setUploadError(isConnErr
              ? 'Lost connection to backend while pipeline was running. Check if the backend is still running.'
              : `Polling error: ${err.message}`
            );
          }
        }
      }, 1500);
      
    } catch (err) {
      stopPolling();
      setIsUploading(false);
      setUploadError(err.message);
    }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setActiveSearch(searchInput.trim());
  };

  const triggerTrace = async (nodeId) => {
    if (graphRef.current) {
      await graphRef.current.traceFunds(nodeId);
    }
  };

  const anomalyNodes = elements.filter(el => el.data && el.data.is_anomaly === true).length;

  return (
    <div className="app-container">
      <div className="top-bar">
        {/* Search Bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Search size={20} color="#94a3b8" />
          <form onSubmit={handleSearch}>
            <input 
              type="text" 
              placeholder="Search Wallet (e.g. W0000191)..." 
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
          </form>
        </div>

        {/* Center: Job Selector & Stats */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          {jobsList.filter(j => j.status === 'completed').length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(15, 23, 42, 0.8)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
              <History size={16} color="#3b82f6" />
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Job:</span>
              <select 
                value={jobId || ''} 
                onChange={handleJobChange}
                style={{ background: 'transparent', color: '#f8fafc', border: 'none', fontSize: '0.85rem', fontWeight: 'bold', outline: 'none', cursor: 'pointer' }}
              >
                {jobsList.filter(j => j.status === 'completed').map(j => (
                  <option key={j.job_id} value={j.job_id} style={{ background: '#0f172a', color: '#fff' }}>
                    {j.job_id}
                  </option>
                ))}
              </select>
            </div>
          )}

          {anomalyNodes > 0 && !isUploading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', padding: '6px 12px', borderRadius: '8px', fontSize: '0.85rem', color: '#fca5a5' }}>
              <AlertTriangle size={14} color="#ef4444" />
              <span><strong>{anomalyNodes}</strong> Flagged Wallets</span>
            </div>
          )}
        </div>
        
        {/* Upload Button */}
        <div className="upload-container">
          <input 
            type="file" 
            accept=".csv" 
            multiple 
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileUpload}
          />
          <button 
            className="btn upload-btn" 
            onClick={() => { setUploadError(null); fileInputRef.current.click(); }} 
            disabled={isUploading}
          >
            <UploadCloud size={16} /> 
            {isUploading ? 'Processing...' : 'Upload 6 CSVs'}
          </button>
        </div>
      </div>

      <div className="graph-container">
        {/* Upload error banner */}
        {uploadError && !isUploading && (
          <div style={{
            position: 'absolute', top: '16px', left: '50%', transform: 'translateX(-50%)',
            zIndex: 1000, background: '#1e1a2e', border: '1px solid #ef4444',
            borderRadius: '12px', padding: '20px 28px', maxWidth: '520px', width: '90%',
            boxShadow: '0 8px 32px rgba(239,68,68,0.25)'
          }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: '12px' }}>
              <XCircle size={22} color="#ef4444" style={{ flexShrink: 0, marginTop: '2px' }} />
              <div style={{ flex: 1 }}>
                <p style={{ fontWeight: 'bold', color: '#fca5a5', marginBottom: '6px' }}>Upload Failed</p>
                <p style={{ fontSize: '0.85rem', color: '#cbd5e1', whiteSpace: 'pre-wrap', lineHeight: '1.5' }}>{uploadError}</p>
              </div>
              <button onClick={() => setUploadError(null)} style={{ background: 'none', border: 'none', color: '#64748b', cursor: 'pointer', flexShrink: 0 }}>
                ✕
              </button>
            </div>
          </div>
        )}

        {isUploading ? (
          <div className="loading-overlay">
            <Loader className="spinner" size={54} color="#3b82f6" />
            <h3 style={{ marginTop: '20px', fontWeight: '600' }}>{statusMessage}</h3>
            <div style={{ width: '360px', height: '8px', background: '#1e293b', borderRadius: '4px', marginTop: '16px', overflow: 'hidden' }}>
              <div style={{ width: `${uploadProgress}%`, height: '100%', background: 'linear-gradient(90deg, #3b82f6, #6366f1)', transition: 'width 0.5s ease', borderRadius: '4px' }} />
            </div>
            <p style={{ marginTop: '10px', fontSize: '0.85rem', color: '#64748b' }}>{uploadProgress}% complete</p>
            <p style={{ marginTop: '6px', fontSize: '0.8rem', color: '#475569' }}>Pipeline stages: Ingestion → Graph → Features → GNN → Fusion → SHAP → Alerts</p>
          </div>
        ) : loading ? (
          <div className="loading-overlay">
            <Loader className="spinner" size={48} color="#3b82f6" />
            <h3 style={{ marginTop: '20px' }}>Loading network visualization...</h3>
          </div>
        ) : elements.length === 0 ? (
          <div style={{ padding: '80px', color: '#94a3b8', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <UploadCloud size={56} style={{ marginBottom: '20px', opacity: 0.4 }} />
            <h2 style={{ marginBottom: '10px' }}>No data loaded</h2>
            <p style={{ maxWidth: '380px', lineHeight: '1.6' }}>
              Upload the 6 Bitcoin CSV files to start the forensic analysis pipeline.
              The backend server must be running on port 8000.
            </p>
            <div style={{ display: 'flex', gap: '12px', marginTop: '24px' }}>
              <button className="btn upload-btn" onClick={() => { setUploadError(null); fileInputRef.current.click(); }}>
                <UploadCloud size={16} style={{ marginRight: '6px', verticalAlign: 'text-bottom' }} />
                Upload 6 CSVs
              </button>
              <button className="btn" style={{ opacity: 0.7 }} onClick={() => window.location.reload()}>
                <RefreshCw size={14} style={{ marginRight: '6px', verticalAlign: 'text-bottom' }} />
                Refresh
              </button>
            </div>
            <code style={{ marginTop: '24px', background: '#0f172a', padding: '10px 16px', borderRadius: '8px', fontSize: '0.8rem', color: '#64748b' }}>
              Required: wallets.csv, transactions.csv, transaction_inputs.csv,<br/>
              transaction_outputs.csv, network_observations.csv, ip_metadata.csv
            </code>
          </div>
        ) : (
          <GraphView 
            ref={graphRef} 
            elements={elements} 
            onNodeSelect={setSelectedNode} 
            searchQuery={activeSearch}
            jobId={jobId}
          />
        )}
      </div>
      
      <Sidebar selectedNode={selectedNode} onTrace={triggerTrace} jobId={jobId} />
    </div>
  );
}
