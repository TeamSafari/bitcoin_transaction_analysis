import React, { useState, useEffect, useRef } from 'react';
import { Search } from 'lucide-react';
import GraphView from './components/GraphView';
import Sidebar from './components/Sidebar';
import { loadGraphData } from './utils/dataParser';

export default function App() {
  const [elements, setElements] = useState([]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [loading, setLoading] = useState(true);
  const [searchInput, setSearchInput] = useState('');
  const [activeSearch, setActiveSearch] = useState('');
  
  const graphRef = useRef();

  useEffect(() => {
    async function initData() {
      try {
        const data = await loadGraphData();
        setElements(data);
      } catch (err) {
        console.error("Error loading CSV files", err);
      } finally {
        setLoading(false);
      }
    }
    initData();
  }, []);

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
        <Search size={20} color="#94a3b8" />
        <form onSubmit={handleSearch}>
            <input 
              type="text" 
              placeholder="Search Wallet ID (e.g. W0000001)..." 
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
            />
        </form>
      </div>

      <div className="graph-container">
        {loading ? <div style={{ padding: '80px', color: '#94a3b8' }}>Loading graph data...</div> : 
          <GraphView ref={graphRef} elements={elements} onNodeSelect={setSelectedNode} searchQuery={activeSearch} />}
      </div>
      
      <Sidebar selectedNode={selectedNode} onTrace={triggerTrace} />
    </div>
  );
}
