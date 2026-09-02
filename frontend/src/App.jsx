import React, { useState, useEffect, useRef } from 'react';
import { Search, UploadCloud, Loader } from 'lucide-react';
import GraphView from './components/GraphView';
import Sidebar from './components/Sidebar';
import { uploadCSV, pollJobStatus, fetchGraphData } from './utils/apiService';

export default function App() {
  const [elements, setElements] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(false);
  
  // Job / Upload State
  const [jobId, setJobId] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  
  const [searchInput, setSearchInput] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  
  const graphRef = useRef();
  const fileInputRef = useRef();

  const handleFileUpload = async (e) => {
      const files = Array.from(e.target.files);
      if (files.length === 0) return;
      
      // We need exactly 6 files
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
          e.target.value = ''; // reset input
          return;
      }
      
      try {
          setIsUploading(true);
          setElements([]); // clear old graph
          setSelectedNode(null);
          setStatusMessage('Uploading 6 CSV files to backend...');
          
          // 1. Upload to backend API
          const uploadRes = await uploadCSV(fileMap);
          const currentJobId = uploadRes.job_id;
          setJobId(currentJobId);
          
          setStatusMessage('Processing ML Pipeline...');
          
          // 2. Poll Status API
          let progress = 0;
          const pollInterval = setInterval(async () => {
              try {
                  const statusRes = await pollJobStatus(currentJobId);
                  progress = statusRes.progress;
                  setStatusMessage(`Processing... ${progress}% - ${statusRes.step || ''}`);
                  
                  // For the mock simulation, we just force it to finish after a few loops
                  if (progress >= 100 || statusRes.status === 'completed' || statusRes.step === 'Complete') {
                      clearInterval(pollInterval);
                      setStatusMessage('Fetching final graph data...');
                      
                      // 3. Fetch Graph API
                      const graphElements = await fetchGraphData(currentJobId);
                      setElements(graphElements);
                      setIsUploading(false);
                  }
              } catch (err) {
                  clearInterval(pollInterval);
                  console.error("Polling error:", err);
                  setStatusMessage('Error during processing. Check backend logs.');
                  setTimeout(() => setIsUploading(false), 4000);
              }
          }, 1500); // poll every 1.5s
          
      } catch (err) {
          console.error("Upload error:", err);
          setStatusMessage('Upload failed. Is the backend running?');
          setTimeout(() => setIsUploading(false), 4000);
      }
  };

  const handleSearch = (e) => {
    e.preventDefault();
    setActiveSearch(searchInput);
  };

  const triggerTrace = (nodeId) => {
    if (graphRef.current) {
        graphRef.current.traceFunds(nodeId);
    }
  };

  return (
    <div className="app-container">
      <div className="top-bar">
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Search size={20} color="#94a3b8" />
            <form onSubmit={handleSearch}>
                <input 
                  type="text" 
                  placeholder="Search Wallet ID..." 
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                />
            </form>
        </div>
        
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
                {isUploading ? 'Processing...' : 'Upload CSV'}
            </button>
        </div>
      </div>

      <div className="graph-container">
        {isUploading ? (
            <div className="loading-overlay">
                <Loader className="spinner" size={48} color="#3b82f6" />
                <h3 style={{ marginTop: '20px' }}>{statusMessage}</h3>
            </div>
        ) : elements.length === 0 ? (
            <div style={{ padding: '80px', color: '#94a3b8', textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
                <UploadCloud size={48} style={{ marginBottom: '20px', opacity: 0.5 }} />
                <h2>No data loaded.</h2>
                <p>Please upload a Bitcoin transaction CSV file to begin analysis.</p>
                <button className="btn upload-btn" style={{ marginTop: '20px' }} onClick={() => fileInputRef.current.click()}>
                    Upload CSV
                </button>
            </div>
        ) : (
            <GraphView ref={graphRef} elements={elements} onNodeSelect={setSelectedNode} searchQuery={activeSearch} />
        )}
      </div>
      
      <Sidebar selectedNode={selectedNode} onTrace={triggerTrace} jobId={jobId} />
    </div>
  );
}
