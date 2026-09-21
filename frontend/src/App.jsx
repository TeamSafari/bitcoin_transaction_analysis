import React, { useState, useEffect, useRef } from 'react';
import { Search, UploadCloud, Loader, AlertTriangle, CheckCircle2, History, ChevronDown } from 'lucide-react';
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
  const [alertsCount, setAlertsCount] = useState(0);
  
  const [searchInput, setSearchInput] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  
  const graphRef = useRef();
  const fileInputRef = useRef();

  // Load existing jobs on mount
  useEffect(() => {
    async function initJobs() {
      try {
        setLoading(true);
        const jobs = await fetchJobsList();
        setJobsList(jobs);

        // Find the latest completed job, prioritizing job-11194 or the first completed job
        const completedJobs = jobs.filter(j => j.status === 'completed');
        let initialJobId = null;

        if (completedJobs.length > 0) {
          const preferred = completedJobs.find(j => j.job_id === 'job-11194') || completedJobs[0];
          initialJobId = preferred.job_id;
        }

        if (initialJobId) {
          setJobId(initialJobId);
          await loadGraphForJob(initialJobId);
        } else {
          // Fallback to default graph overview
          const defaultElements = await fetchGraphData(null);
          setElements(defaultElements);
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
      
      const alertsData = await fetchAlerts(selectedJobId);
      setAlertsCount((alertsData.alerts || []).length);
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

  const handleFileUpload = async (e) => {
      const files = Array.from(e.target.files);
      if (files.length === 0) return;
      
      const fileMap = {};
      const requiredKeys = ['wallets', 'transactions', 'transaction_inputs', 'transaction_outputs', 'network_observations', 'ip_metadata'];
      
      files.forEach(f => {
          const name = f.name.toLowerCase();
          requiredKeys.forEach(k => {
              if (name.includes(k)) fileMap[k] = f;
          });
      });
      
      const missing = requiredKeys.filter(k => !fileMap[k]);
      if (missing.length > 0) {
          alert(`Please select all 6 required CSV files at once.\nMissing: ${missing.join(', ')}`);
          e.target.value = '';
          return;
      }
      
      try {
          setIsUploading(true);
          setUploadProgress(5);
          setElements([]);
          setSelectedNode(null);
          setStatusMessage('Uploading 6 CSV files to forensic backend...');
          
          // 1. Upload to backend API
          const uploadRes = await uploadCSV(fileMap);
          const currentJobId = uploadRes.job_id;
          setJobId(currentJobId);
          
          setStatusMessage('Ingesting data & running Feature Engineering...');
          
          // 2. Poll Status API
          const pollInterval = setInterval(async () => {
              try {
                  const statusRes = await pollJobStatus(currentJobId);
                  const progress = statusRes.progress || 0;
                  setUploadProgress(progress);
                  setStatusMessage(`Processing ML Pipeline: ${progress}% - ${statusRes.step || ''}`);
                  
                  if (progress >= 100 || statusRes.status === 'completed') {
                      clearInterval(pollInterval);
                      setStatusMessage('Rendering forensic network graph...');
                      
                      // Refresh jobs list
                      const updatedJobs = await fetchJobsList();
                      setJobsList(updatedJobs);

                      // 3. Fetch Graph API
                      await loadGraphForJob(currentJobId);
                      setIsUploading(false);
                  } else if (statusRes.status === 'failed') {
                      clearInterval(pollInterval);
                      setStatusMessage(`Job failed: ${statusRes.error_message || 'Check backend logs'}`);
                      setTimeout(() => setIsUploading(false), 4000);
                  }
              } catch (err) {
                  clearInterval(pollInterval);
                  console.error("Polling error:", err);
                  setStatusMessage('Error polling status. Check backend connection.');
                  setTimeout(() => setIsUploading(false), 4000);
              }
          }, 1500);
          
      } catch (err) {
          console.error("Upload error:", err);
          setStatusMessage('Upload failed. Ensure backend API is running on port 8000.');
          setTimeout(() => setIsUploading(false), 4000);
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
            {jobsList.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', background: 'rgba(15, 23, 42, 0.8)', padding: '6px 12px', borderRadius: '8px', border: '1px solid var(--glass-border)' }}>
                    <History size={16} color="#3b82f6" />
                    <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>Job:</span>
                    <select 
                      value={jobId || ''} 
                      onChange={handleJobChange}
                      style={{ background: 'transparent', color: '#f8fafc', border: 'none', fontSize: '0.85rem', fontWeight: 'bold', outline: 'none', cursor: 'pointer' }}
                    >
                      {jobsList.map(j => (
                        <option key={j.job_id} value={j.job_id} style={{ background: '#0f172a', color: '#fff' }}>
                          {j.job_id} ({j.status})
                        </option>
                      ))}
                    </select>
                </div>
            )}

            {alertsCount > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: 'rgba(239, 68, 68, 0.15)', border: '1px solid rgba(239, 68, 68, 0.4)', padding: '6px 12px', borderRadius: '8px', fontSize: '0.85rem', color: '#fca5a5' }}>
                    <AlertTriangle size={14} color="#ef4444" />
                    <span><strong>{alertsCount}</strong> Anomalies</span>
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
            <button className="btn upload-btn" onClick={() => fileInputRef.current.click()} disabled={isUploading}>
                <UploadCloud size={16} /> 
                {isUploading ? 'Processing...' : 'Upload 6 CSVs'}
            </button>
        </div>
      </div>

      <div className="graph-container">
        {isUploading ? (
            <div className="loading-overlay">
                <Loader className="spinner" size={54} color="#3b82f6" />
                <h3 style={{ marginTop: '20px', fontWeight: '600' }}>{statusMessage}</h3>
                <div style={{ width: '320px', height: '8px', background: '#1e293b', borderRadius: '4px', marginTop: '16px', overflow: 'hidden' }}>
                  <div style={{ width: `${uploadProgress}%`, height: '100%', background: '#3b82f6', transition: 'width 0.3s ease' }} />
                </div>
            </div>
        ) : loading ? (
            <div className="loading-overlay">
                <Loader className="spinner" size={48} color="#3b82f6" />
                <h3 style={{ marginTop: '20px' }}>Loading network visualization...</h3>
            </div>
        ) : elements.length === 0 ? (
            <div style={{ padding: '80px', color: '#94a3b8', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <UploadCloud size={48} style={{ marginBottom: '20px', opacity: 0.5 }} />
                <h2>No data loaded.</h2>
                <p>Please upload the 6 Bitcoin CSV files to begin forensic analysis.</p>
                <button className="btn upload-btn" style={{ marginTop: '20px' }} onClick={() => fileInputRef.current.click()}>
                    Upload 6 CSVs
                </button>
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
