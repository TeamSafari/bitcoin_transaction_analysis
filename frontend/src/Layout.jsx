import React, { useState, useEffect, useCallback } from 'react';
import { NavLink, Outlet, useLocation } from 'react-router-dom';
import { Network, LayoutDashboard, Database, Settings, Activity, Moon, Sun } from 'lucide-react';
import { fetchJobsList, fetchGraphData, fetchAlerts } from './utils/apiService';

export default function Layout() {
  const location = useLocation();

  // ── Global state ──────────────────────────────────────────
  const [globalJobId, setGlobalJobId] = useState(null);
  const [jobsList, setJobsList] = useState([]);

  // ── Cached data (fetched once per job, shared across all pages) ──
  const [cachedGraphElements, setCachedGraphElements] = useState([]);
  const [cachedAlerts, setCachedAlerts] = useState([]);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataReady, setDataReady] = useState(false);

  // ── Dark mode state ───────────────────────────────────────
  const [isDarkMode, setIsDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'true';
  });

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add('dark');
      localStorage.setItem('darkMode', 'true');
    } else {
      document.body.classList.remove('dark');
      localStorage.setItem('darkMode', 'false');
    }
  }, [isDarkMode]);

  // ── Refresh jobs list ─────────────────────────────────────
  const refreshJobs = useCallback(async () => {
    const jobs = await fetchJobsList();
    setJobsList(jobs);
    return jobs;
  }, []);

  // ── On first mount, load jobs list ────────────────────────
  useEffect(() => {
    refreshJobs();
  }, [refreshJobs]);

  // ── Whenever globalJobId changes, fetch data ONCE and cache it ──
  useEffect(() => {
    if (!globalJobId) {
      setCachedGraphElements([]);
      setCachedAlerts([]);
      setDataReady(false);
      return;
    }

    let cancelled = false;

    async function loadJobData() {
      setDataLoading(true);
      try {
        const [elements, alertsData] = await Promise.all([
          fetchGraphData(globalJobId).catch(() => []),
          fetchAlerts(globalJobId).catch(() => ({ alerts: [] })),
        ]);
        if (!cancelled) {
          setCachedGraphElements(elements);
          setCachedAlerts(alertsData.alerts || []);
          setDataReady(true);
        }
      } catch (err) {
        console.error('Failed to load job data:', err);
      } finally {
        if (!cancelled) setDataLoading(false);
      }
    }

    loadJobData();
    return () => { cancelled = true; };
  }, [globalJobId]);

  // ── Context shared with ALL child pages via <Outlet context={...}> ──
  const ctx = {
    globalJobId,
    setGlobalJobId,
    jobsList,
    refreshJobs,
    cachedGraphElements,
    setCachedGraphElements,
    cachedAlerts,
    setCachedAlerts,
    dataLoading,
    dataReady,
    isDarkMode,
  };

  return (
    <div className="app-shell">
      <nav className="nav-sidebar">
        <div className="nav-logo">
          <h1>⛓ Bitcoin Forensics</h1>
          <p>Graph-based ML detection</p>
        </div>

        <div className="nav-actions" style={{ marginTop: '10px' }}>
          <div className="nav-section-label">Main Menu</div>
          <NavLink to="/" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} style={({ isActive }) => isActive ? { background: 'rgba(79,142,247,0.15)', color: 'var(--accent)' } : {}}>
            <LayoutDashboard size={15} /> Dashboard
          </NavLink>
          <NavLink to="/graph" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} style={({ isActive }) => isActive ? { background: 'rgba(79,142,247,0.15)', color: 'var(--accent)' } : {}}>
            <Network size={15} /> Network Graph
          </NavLink>
          <NavLink to="/records" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} style={({ isActive }) => isActive ? { background: 'rgba(79,142,247,0.15)', color: 'var(--accent)' } : {}}>
            <Database size={15} /> Records
          </NavLink>
          <NavLink to="/simulator" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} style={({ isActive }) => isActive ? { background: 'rgba(79,142,247,0.15)', color: 'var(--accent)' } : {}}>
            <Activity size={15} /> Simulator
          </NavLink>
        </div>

        <div style={{ flex: 1 }} />

        <div className="nav-actions" style={{ borderTop: '1px solid var(--border)', borderBottom: 'none' }}>
          <button 
            className="nav-btn ghost" 
            onClick={() => setIsDarkMode(prev => !prev)}
            style={{ width: '100%', textAlign: 'left', padding: '9px 12px' }}
          >
            {isDarkMode ? <Sun size={15} /> : <Moon size={15} />}
            {isDarkMode ? 'Light Mode' : 'Dark Mode'}
          </button>
          <NavLink to="/settings" className={({ isActive }) => `nav-btn ${isActive ? 'active' : ''}`} style={({ isActive }) => isActive ? { background: 'rgba(79,142,247,0.15)', color: 'var(--accent)' } : {}}>
            <Settings size={15} /> Settings
          </NavLink>
        </div>
      </nav>

      <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
        <Outlet context={ctx} />
      </div>
    </div>
  );
}
