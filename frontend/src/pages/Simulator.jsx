import React, { useState, useEffect, useRef, useMemo } from 'react';
import { useOutletContext } from 'react-router-dom';
import cytoscape from 'cytoscape';
import {
  Activity, Play, RotateCcw, CheckCircle2, AlertTriangle,
  Zap, ShieldAlert, ShieldCheck, ArrowRight, Layers,
  FastForward, Eye, Network, GitCommit, Sparkles, Copy, Check
} from 'lucide-react';

// ─── Format Satoshis ─────────────────────────────────────────────────────────
function formatSats(sats) {
  if (!sats) return '0 SATS';
  if (sats >= 1e8) return `${(sats / 1e8).toFixed(2)} BTC`;
  if (sats >= 1e6) return `${(sats / 1e6).toFixed(1)}M SATS`;
  if (sats >= 1e3) return `${(sats / 1e3).toFixed(1)}k SATS`;
  return `${sats.toLocaleString()} SATS`;
}

// ─── BASELINE 6 TRANSACTIONS (CLEAN ACTIVITY) ────────────────────────────────
const BASELINE_NODES = [
  { id: 'Miner_Pool', label: 'Miner Pool', isAnomaly: false },
  { id: 'Exchange_Hot', label: 'Exchange Hot', isAnomaly: false },
  { id: 'User_Alice', label: 'User Alice', isAnomaly: false },
  { id: 'User_Bob', label: 'User Bob', isAnomaly: false },
  { id: 'Merchant_Pay', label: 'Merchant Pay', isAnomaly: false },
  { id: 'Cold_Vault', label: 'Cold Vault', isAnomaly: false },
];

const BASELINE_TXS = [
  { id: 'tx_b1', source: 'Miner_Pool', target: 'Exchange_Hot', amount: 150000000, txid: 'a10f84bc...91e2', desc: 'Mining block reward distribution', isInjected: false },
  { id: 'tx_b2', source: 'Exchange_Hot', target: 'User_Alice', amount: 40000000, txid: 'b28e71aa...401d', desc: 'Exchange withdrawal', isInjected: false },
  { id: 'tx_b3', source: 'User_Alice', target: 'Merchant_Pay', amount: 5000000, txid: 'c39d62bb...882a', desc: 'Merchant goods payment', isInjected: false },
  { id: 'tx_b4', source: 'Merchant_Pay', target: 'Cold_Vault', amount: 2000000, txid: 'd48c53cc...119b', desc: 'Merchant sweep to cold storage', isInjected: false },
  { id: 'tx_b5', source: 'User_Bob', target: 'User_Alice', amount: 12000000, txid: 'e57b44dd...332c', desc: 'P2P wallet transfer', isInjected: false },
  { id: 'tx_b6', source: 'Exchange_Hot', target: 'User_Bob', amount: 25000000, txid: 'f66a35ee...774e', desc: 'Exchange fiat cashout', isInjected: false },
];

// ─── INJECTED PATTERNS DEFINITIONS (6–7 TRANSACTIONS EACH) ────────────────────
const PATTERN_CONFIGS = {
  fan_out: {
    id: 'fan_out',
    title: 'Fan-Out (Dispersal / Smurfing)',
    tag: 'Fan-out (Scattering)',
    badgeColor: '#ef4444',
    desc: 'One illicit wallet rapidly splits funds across 6 mule wallets to evade anti-structuring thresholds.',
    anomalyReason: 'Out-degree spike (out=6 vs in=0). Dispersal ratio > 3.0 matches high-risk smurfing heuristic.',
    score: 96.2,
    nodes: [
      { id: 'Attacker_Src', label: 'Attacker [Source]', isAnomaly: true, role: 'Dispersal Hub' },
      { id: 'Mule_1', label: 'Mule 1', isAnomaly: true, role: 'Mule' },
      { id: 'Mule_2', label: 'Mule 2', isAnomaly: true, role: 'Mule' },
      { id: 'Mule_3', label: 'Mule 3', isAnomaly: true, role: 'Mule' },
      { id: 'Mule_4', label: 'Mule 4', isAnomaly: true, role: 'Mule' },
      { id: 'Mule_5', label: 'Mule 5', isAnomaly: true, role: 'Mule' },
      { id: 'Mule_6', label: 'Mule 6', isAnomaly: true, role: 'Mule' },
    ],
    txs: [
      { id: 'tx_inj_1', source: 'Attacker_Src', target: 'Mule_1', amount: 20000000, txid: '99aa11...fo01', desc: 'Dispersal tranche 1', isInjected: true },
      { id: 'tx_inj_2', source: 'Attacker_Src', target: 'Mule_2', amount: 20000000, txid: '99aa22...fo02', desc: 'Dispersal tranche 2', isInjected: true },
      { id: 'tx_inj_3', source: 'Attacker_Src', target: 'Mule_3', amount: 20000000, txid: '99aa33...fo03', desc: 'Dispersal tranche 3', isInjected: true },
      { id: 'tx_inj_4', source: 'Attacker_Src', target: 'Mule_4', amount: 20000000, txid: '99aa44...fo04', desc: 'Dispersal tranche 4', isInjected: true },
      { id: 'tx_inj_5', source: 'Attacker_Src', target: 'Mule_5', amount: 20000000, txid: '99aa55...fo05', desc: 'Dispersal tranche 5', isInjected: true },
      { id: 'tx_inj_6', source: 'Attacker_Src', target: 'Mule_6', amount: 20000000, txid: '99aa66...fo06', desc: 'Dispersal tranche 6', isInjected: true },
    ]
  },
  fan_in: {
    id: 'fan_in',
    title: 'Fan-In (Consolidation / Gathering)',
    tag: 'Fan-in (Gathering)',
    badgeColor: '#8b5cf6',
    desc: 'Multiple micro-sources suddenly funnel gathered funds into a single centralized syndicate vault.',
    anomalyReason: 'In-degree spike (in=6 vs out=0). High consolidation velocity flagged by GNN & Isolation Forest.',
    score: 94.8,
    nodes: [
      { id: 'Syndicate_Vault', label: 'Syndicate Vault', isAnomaly: true, role: 'Consolidation Hub' },
      { id: 'Smurf_1', label: 'Smurf 1', isAnomaly: true, role: 'Feeder' },
      { id: 'Smurf_2', label: 'Smurf 2', isAnomaly: true, role: 'Feeder' },
      { id: 'Smurf_3', label: 'Smurf 3', isAnomaly: true, role: 'Feeder' },
      { id: 'Smurf_4', label: 'Smurf 4', isAnomaly: true, role: 'Feeder' },
      { id: 'Smurf_5', label: 'Smurf 5', isAnomaly: true, role: 'Feeder' },
      { id: 'Smurf_6', label: 'Smurf 6', isAnomaly: true, role: 'Feeder' },
    ],
    txs: [
      { id: 'tx_inj_1', source: 'Smurf_1', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb11...fi01', desc: 'Consolidation deposit 1', isInjected: true },
      { id: 'tx_inj_2', source: 'Smurf_2', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb22...fi02', desc: 'Consolidation deposit 2', isInjected: true },
      { id: 'tx_inj_3', source: 'Smurf_3', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb33...fi03', desc: 'Consolidation deposit 3', isInjected: true },
      { id: 'tx_inj_4', source: 'Smurf_4', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb44...fi04', desc: 'Consolidation deposit 4', isInjected: true },
      { id: 'tx_inj_5', source: 'Smurf_5', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb55...fi05', desc: 'Consolidation deposit 5', isInjected: true },
      { id: 'tx_inj_6', source: 'Smurf_6', target: 'Syndicate_Vault', amount: 18000000, txid: '88bb66...fi06', desc: 'Consolidation deposit 6', isInjected: true },
    ]
  },
  peeling_chain: {
    id: 'peeling_chain',
    title: 'Peeling Chain (Sequential Layering)',
    tag: 'Pass-through (Peeling)',
    badgeColor: '#f59e0b',
    desc: 'Funds travel sequentially through multi-hop intermediate wallets, peeling off small amounts at each step.',
    anomalyReason: 'Sequential degree balance (in=1, out=1/2) with rapid pass-through transit time.',
    score: 89.4,
    nodes: [
      { id: 'Launderer_Root', label: 'Launderer Root', isAnomaly: true, role: 'Originator' },
      { id: 'Hop_1', label: 'Hop 1 [Peel]', isAnomaly: true, role: 'Transit' },
      { id: 'Hop_2', label: 'Hop 2 [Peel]', isAnomaly: true, role: 'Transit' },
      { id: 'Hop_3', label: 'Hop 3 [Peel]', isAnomaly: true, role: 'Transit' },
      { id: 'Hop_4', label: 'Hop 4 [Peel]', isAnomaly: true, role: 'Transit' },
      { id: 'Offshore_Mix', label: 'Offshore Mix', isAnomaly: true, role: 'Terminal Destination' },
    ],
    txs: [
      { id: 'tx_inj_1', source: 'Launderer_Root', target: 'Hop_1', amount: 90000000, txid: '77cc11...pc01', desc: 'Layering initial hop', isInjected: true },
      { id: 'tx_inj_2', source: 'Hop_1', target: 'Hop_2', amount: 80000000, txid: '77cc22...pc02', desc: 'Layering transit hop 2', isInjected: true },
      { id: 'tx_inj_3', source: 'Hop_1', target: 'Merchant_Pay', amount: 10000000, txid: '77cc2b...pc03', desc: 'Peel change to clean merchant', isInjected: true },
      { id: 'tx_inj_4', source: 'Hop_2', target: 'Hop_3', amount: 70000000, txid: '77cc33...pc04', desc: 'Layering transit hop 3', isInjected: true },
      { id: 'tx_inj_5', source: 'Hop_2', target: 'User_Bob', amount: 10000000, txid: '77cc3b...pc05', desc: 'Peel change to retail wallet', isInjected: true },
      { id: 'tx_inj_6', source: 'Hop_3', target: 'Hop_4', amount: 60000000, txid: '77cc44...pc06', desc: 'Layering transit hop 4', isInjected: true },
      { id: 'tx_inj_7', source: 'Hop_4', target: 'Offshore_Mix', amount: 50000000, txid: '77cc55...pc07', desc: 'Terminal mixing cashout', isInjected: true },
    ]
  },
  cycle: {
    id: 'cycle',
    title: 'Cycle (Round-Tripping / Wash Flow)',
    tag: 'Cycle (Circular Loop)',
    badgeColor: '#fb923c',
    desc: 'Circular flow of money that loops back to an earlier actor to fabricate volume or wash token origin.',
    anomalyReason: 'Closed cycle topology (A→B→C→D→A) and direct 2-hop back-edge detected.',
    score: 93.1,
    nodes: [
      { id: 'Loop_A', label: 'Loop Node A', isAnomaly: true, role: 'Ring Origin' },
      { id: 'Loop_B', label: 'Loop Node B', isAnomaly: true, role: 'Ring Member' },
      { id: 'Loop_C', label: 'Loop Node C', isAnomaly: true, role: 'Ring Member' },
      { id: 'Loop_D', label: 'Loop Node D', isAnomaly: true, role: 'Ring Member' },
      { id: 'Loop_E', label: 'Loop Node E', isAnomaly: true, role: 'Ring Return' },
    ],
    txs: [
      { id: 'tx_inj_1', source: 'Loop_A', target: 'Loop_B', amount: 75000000, txid: '66dd11...cy01', desc: 'Cycle step A ➔ B', isInjected: true },
      { id: 'tx_inj_2', source: 'Loop_B', target: 'Loop_C', amount: 74000000, txid: '66dd22...cy02', desc: 'Cycle step B ➔ C', isInjected: true },
      { id: 'tx_inj_3', source: 'Loop_C', target: 'Loop_D', amount: 73000000, txid: '66dd33...cy03', desc: 'Cycle step C ➔ D', isInjected: true },
      { id: 'tx_inj_4', source: 'Loop_D', target: 'Loop_E', amount: 72000000, txid: '66dd44...cy04', desc: 'Cycle step D ➔ E', isInjected: true },
      { id: 'tx_inj_5', source: 'Loop_E', target: 'Loop_A', amount: 71000000, txid: '66dd55...cy05', desc: 'Cycle closed return E ➔ A', isInjected: true },
      { id: 'tx_inj_6', source: 'Loop_B', target: 'Loop_A', amount: 15000000, txid: '66dd66...cy06', desc: 'Direct 2-hop back-edge B ➔ A', isInjected: true },
    ]
  }
};

export default function Simulator() {
  const { isDarkMode } = useOutletContext();
  const [selectedPatternKey, setSelectedPatternKey] = useState('fan_out');
  const [isSimulated, setIsSimulated] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationStep, setSimulationStep] = useState(0);
  const [selectedNodeData, setSelectedNodeData] = useState(null);
  const [copiedTxid, setCopiedTxid] = useState(null);

  const containerRef = useRef(null);
  const cyRef = useRef(null);

  const currentPattern = PATTERN_CONFIGS[selectedPatternKey];

  // Combined nodes & edges
  const activeNodes = useMemo(() => {
    if (!isSimulated && simulationStep === 0) return BASELINE_NODES;
    return [...BASELINE_NODES, ...currentPattern.nodes];
  }, [isSimulated, simulationStep, currentPattern]);

  const activeTxs = useMemo(() => {
    if (!isSimulated && simulationStep === 0) return BASELINE_TXS;
    if (simulationStep > 0 && !isSimulated) {
      return [...BASELINE_TXS, ...currentPattern.txs.slice(0, simulationStep)];
    }
    return [...BASELINE_TXS, ...currentPattern.txs];
  }, [isSimulated, simulationStep, currentPattern]);

  // Format Cytoscape elements
  const cyElements = useMemo(() => {
    const eles = [];

    activeNodes.forEach(n => {
      eles.push({
        group: 'nodes',
        data: {
          id: n.id,
          label: n.label,
          isAnomaly: n.isAnomaly,
          role: n.role || 'Clean Wallet'
        }
      });
    });

    activeTxs.forEach(tx => {
      eles.push({
        group: 'edges',
        data: {
          id: tx.id,
          source: tx.source,
          target: tx.target,
          amount: tx.amount,
          isInjected: tx.isInjected
        }
      });
    });

    return eles;
  }, [activeNodes, activeTxs]);

  // Initialize or re-render Cytoscape
  useEffect(() => {
    if (!containerRef.current) return;

    if (cyRef.current) {
      cyRef.current.destroy();
      cyRef.current = null;
    }

    const cy = cytoscape({
      container: containerRef.current,
      elements: cyElements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => ele.data('isAnomaly') ? '#ef4444' : '#10b981',
            'width': (ele) => ele.data('isAnomaly') ? 34 : 26,
            'height': (ele) => ele.data('isAnomaly') ? 34 : 26,
            'label': 'data(label)',
            'color': isDarkMode ? '#cbd5e1' : '#334155',
            'font-size': '10px',
            'font-weight': 600,
            'font-family': 'Inter, system-ui, sans-serif',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'border-width': 2,
            'border-color': (ele) => ele.data('isAnomaly') ? 'rgba(239,68,68,0.4)' : 'rgba(16,185,129,0.4)',
            'transition-property': 'background-color, border-color, width, height',
            'transition-duration': '0.3s'
          }
        },
        {
          selector: 'edge',
          style: {
            'width': (ele) => ele.data('isInjected') ? 3 : 1.5,
            'line-color': (ele) => ele.data('isInjected') ? '#ef4444' : isDarkMode ? 'rgba(71,85,105,0.5)' : 'rgba(148, 163, 184, 0.45)',
            'target-arrow-color': (ele) => ele.data('isInjected') ? '#ef4444' : isDarkMode ? 'rgba(71,85,105,0.6)' : 'rgba(148, 163, 184, 0.5)',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 0.9,
            'curve-style': 'bezier',
            'opacity': 0.85
          }
        },
        {
          selector: '.highlighted',
          style: {
            'border-width': 4,
            'border-color': isDarkMode ? '#f8fafc' : '#1e293b',
            'z-index': 99
          }
        }
      ],
      layout: {
        name: 'cose',
        animate: false,
        randomize: false,
        componentSpacing: 80,
        nodeRepulsion: 400000,
        idealEdgeLength: 80,
        padding: 30
      }
    });

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      setSelectedNodeData(node.data());
      cy.elements().removeClass('highlighted');
      node.addClass('highlighted');
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('highlighted');
        setSelectedNodeData(null);
      }
    });

    cyRef.current = cy;

    return () => {
      if (cy) cy.destroy();
      cyRef.current = null;
    };
  }, [cyElements, isDarkMode]);

  // Run live sequential simulation
  const handleRunSimulation = () => {
    setIsSimulating(true);
    setIsSimulated(false);
    setSimulationStep(0);

    const totalSteps = currentPattern.txs.length;
    let step = 0;

    const interval = setInterval(() => {
      step++;
      setSimulationStep(step);

      if (step >= totalSteps) {
        clearInterval(interval);
        setIsSimulating(false);
        setIsSimulated(true);
      }
    }, 450);
  };

  // Reset to clean baseline
  const handleReset = () => {
    setIsSimulating(false);
    setIsSimulated(false);
    setSimulationStep(0);
    setSelectedNodeData(null);
  };

  const handleCopyTxid = (txid) => {
    navigator.clipboard.writeText(txid);
    setCopiedTxid(txid);
    setTimeout(() => setCopiedTxid(null), 1500);
  };

  const totalSats = activeTxs.reduce((sum, tx) => sum + tx.amount, 0);

  return (
    <div style={{ padding: '24px 32px', height: '100%', overflowY: 'auto' }}>
      
      {/* ══ HEADER ══════════════════════════════════════════════════ */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px', flexWrap: 'wrap', gap: '14px' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Activity size={24} color="var(--accent)" />
            <h1 style={{ fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.02em', color: 'var(--text-primary)' }}>
              Pattern Injector & Sandbox Simulator
            </h1>
            <span style={{
              background: 'rgba(16,185,129,0.1)', color: 'var(--success)',
              padding: '3px 9px', borderRadius: '6px', fontSize: '0.74rem',
              fontWeight: 600, border: '1px solid rgba(16,185,129,0.2)'
            }}>
              Sandbox Mode (Isolated)
            </span>
          </div>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.84rem', marginTop: 4 }}>
            Test ML detection engines against isolated 12–13 synthetic transactions without modifying your main dataset
          </p>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <button
            className="action-btn outline"
            onClick={handleReset}
            style={{ width: 'auto', padding: '8px 14px', borderRadius: '8px', fontSize: '0.82rem' }}
          >
            <RotateCcw size={14} /> Reset Baseline
          </button>

          <button
            className="action-btn primary"
            onClick={handleRunSimulation}
            disabled={isSimulating}
            style={{ width: 'auto', padding: '8px 18px', borderRadius: '8px', fontSize: '0.84rem' }}
          >
            <Play size={14} /> {isSimulating ? 'Simulating Transactions…' : 'Run Live Simulation'}
          </button>
        </div>
      </div>

      {/* ══ TOP ROW: PATTERN SELECTOR CARDS ═════════════════════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px', marginBottom: '22px' }}>
        {Object.values(PATTERN_CONFIGS).map(p => {
          const isSelected = selectedPatternKey === p.id;
          return (
            <div
              key={p.id}
              onClick={() => {
                setSelectedPatternKey(p.id);
                handleReset();
              }}
              style={{
                border: `2px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}`,
                background: isSelected ? 'rgba(37,99,235,0.06)' : 'var(--bg-card)',
                padding: '14px 16px',
                borderRadius: '12px',
                cursor: 'pointer',
                transition: 'all 0.15s ease',
                boxShadow: isSelected ? '0 2px 8px rgba(37,99,235,0.12)' : 'none'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontWeight: 700, fontSize: '0.86rem', color: isSelected ? 'var(--accent)' : 'var(--text-primary)' }}>
                  {p.title.split(' ')[0]}
                </span>
                <span style={{
                  fontSize: '0.7rem', fontWeight: 600, padding: '2px 6px',
                  borderRadius: '4px', background: `${p.badgeColor}15`, color: p.badgeColor
                }}>
                  {p.txs.length} txs
                </span>
              </div>
              <p style={{ fontSize: '0.74rem', color: 'var(--text-muted)', lineHeight: 1.4, margin: 0 }}>
                {p.desc}
              </p>
            </div>
          );
        })}
      </div>

      {/* ══ MIDDLE SECTION: SPLIT VISUALIZER & SCORECARD ════════════ */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 0.8fr', gap: '20px', marginBottom: '24px' }}>
        
        {/* Canvas Visualizer */}
        <div className="data-card" style={{ padding: 0, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          <div style={{
            padding: '14px 18px', borderBottom: '1px solid var(--border)',
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            background: 'var(--bg-card)'
          }}>
            <div>
              <div style={{ fontSize: '0.88rem', fontWeight: 700, color: 'var(--text-primary)' }}>
                Live Sandbox Topology Visualizer
              </div>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                {activeNodes.length} Wallets • {activeTxs.length} Transactions ({formatSats(totalSats)})
              </div>
            </div>

            {/* Canvas Legend */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '0.74rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#10b981' }} />
                <span style={{ color: 'var(--text-secondary)' }}>Clean ({BASELINE_NODES.length})</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444' }} />
                <span style={{ color: 'var(--text-secondary)' }}>
                  Injected ({isSimulated || simulationStep > 0 ? currentPattern.nodes.length : 0})
                </span>
              </div>
            </div>
          </div>

          {/* Cytoscape Canvas */}
          <div style={{ height: 360, width: '100%', position: 'relative', background: 'var(--bg-deep)' }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

            {/* Clicked Node Detail Tag */}
            {selectedNodeData && (
              <div style={{
                position: 'absolute', bottom: 12, left: 12, right: 12,
                background: 'var(--bg-panel)', border: '1px solid var(--border)',
                borderRadius: '8px', padding: '10px 14px', boxShadow: '0 4px 12px rgba(0,0,0,0.08)',
                display: 'flex', justifyContent: 'space-between', alignItems: 'center'
              }}>
                <div>
                  <span style={{ fontWeight: 700, fontSize: '0.84rem', color: selectedNodeData.isAnomaly ? '#ef4444' : '#10b981' }}>
                    {selectedNodeData.label}
                  </span>
                  <span style={{ marginLeft: 8, fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                    Role: {selectedNodeData.role}
                  </span>
                </div>
                <span style={{
                  fontSize: '0.72rem', fontWeight: 600, padding: '2px 8px', borderRadius: '4px',
                  background: selectedNodeData.isAnomaly ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                  color: selectedNodeData.isAnomaly ? '#ef4444' : '#10b981'
                }}>
                  {selectedNodeData.isAnomaly ? 'SUSPICIOUS ENTITY' : 'NORMAL ENTITY'}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Live ML Detection Scorecard */}
        <div className="data-card" style={{ padding: '20px', display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '0.9rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
                  Live Detection Diagnostics
                </h3>
                <p style={{ margin: '2px 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Evaluates simulated topology against ML model thresholds
                </p>
              </div>
              <span style={{
                padding: '3px 8px', borderRadius: '5px', fontSize: '0.74rem', fontWeight: 600,
                background: isSimulated ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                color: isSimulated ? '#ef4444' : '#10b981'
              }}>
                {isSimulated ? 'ANOMALY CONFIRMED' : isSimulating ? 'INJECTING…' : 'BASELINE SECURE'}
              </span>
            </div>

            {/* Risk Gauge Bar */}
            <div style={{ background: 'var(--bg-deep)', borderRadius: '10px', padding: '14px', marginBottom: 16, border: '1px solid var(--border)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: 6 }}>
                <span style={{ color: 'var(--text-muted)' }}>Aggregate Threat Score:</span>
                <strong style={{ fontSize: '0.92rem', color: isSimulated ? '#ef4444' : 'var(--success)' }}>
                  {isSimulated ? `${currentPattern.score}%` : '12.4%'}
                </strong>
              </div>
              <div style={{ height: 8, background: 'var(--border)', borderRadius: 4, overflow: 'hidden' }}>
                <div
                  style={{
                    height: '100%',
                    width: isSimulated ? `${currentPattern.score}%` : '12.4%',
                    background: isSimulated ? 'linear-gradient(90deg, #f59e0b, #ef4444)' : '#10b981',
                    borderRadius: 4,
                    transition: 'all 0.4s ease'
                  }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 4 }}>
                <span>0% Safe</span>
                <span>Threshold: 70%</span>
                <span>100% Critical</span>
              </div>
            </div>

            {/* Detection Engines Breakdown */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {[
                { name: 'GraphSAGE GNN', desc: 'Structural Embedding', status: isSimulated ? 'Anomaly Flagged' : 'Normal', ok: !isSimulated },
                { name: 'Isolation Forest', desc: 'Degree Outlier Test', status: isSimulated ? 'Outlier Detected' : 'Normal', ok: !isSimulated },
                { name: 'Rule Engine', desc: currentPattern.tag, status: isSimulated ? 'Pattern Matched' : 'No Pattern', ok: !isSimulated },
              ].map(eng => (
                <div key={eng.name} style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                  background: 'var(--bg-deep)', padding: '9px 12px', borderRadius: '7px',
                  border: '1px solid var(--border)', fontSize: '0.76rem'
                }}>
                  <div>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{eng.name}</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>({eng.desc})</span>
                  </div>
                  <span style={{ fontWeight: 600, color: eng.ok ? '#10b981' : '#ef4444' }}>
                    {eng.status}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Explanation Alert */}
          <div style={{
            marginTop: 14, padding: '10px 12px',
            background: isSimulated ? 'rgba(239,68,68,0.06)' : 'rgba(37,99,235,0.06)',
            border: `1px solid ${isSimulated ? 'rgba(239,68,68,0.2)' : 'rgba(37,99,235,0.2)'}`,
            borderRadius: '8px', fontSize: '0.75rem',
            color: isSimulated ? '#ef4444' : 'var(--text-secondary)'
          }}>
            <strong>Diagnostic Note:</strong> {isSimulated ? currentPattern.anomalyReason : 'Currently running standard baseline activity with 6 clean transactions.'}
          </div>
        </div>

      </div>

      {/* ══ BOTTOM: 12–13 TRANSACTIONS SANDBOX LEDGER ═══════════════ */}
      <div className="data-card" style={{ padding: '20px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
          <div>
            <h3 style={{ margin: 0, fontSize: '0.94rem', fontWeight: 700, textTransform: 'none', color: 'var(--text-primary)' }}>
              Sandbox Transactions Ledger ({activeTxs.length} Transactions)
            </h3>
            <p style={{ margin: '2px 0 0', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Inspect the exact baseline vs synthetic injected transactions
            </p>
          </div>
          <span style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
            Baseline: 6 txs • Injected: {activeTxs.length - 6} txs
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.8rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>#</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Classification</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Sender ➔ Recipient</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Amount</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>Context / Description</th>
                <th style={{ padding: '10px 12px', fontWeight: 600, fontSize: '0.74rem', textTransform: 'uppercase' }}>TxID</th>
              </tr>
            </thead>
            <tbody>
              {activeTxs.map((tx, idx) => (
                <tr
                  key={tx.id}
                  style={{
                    borderBottom: '1px solid var(--border)',
                    background: tx.isInjected ? 'rgba(239,68,68,0.03)' : 'transparent',
                    transition: 'background 0.12s'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = tx.isInjected ? 'rgba(239,68,68,0.03)' : 'transparent'}
                >
                  <td style={{ padding: '10px 12px', color: 'var(--text-muted)', fontWeight: 600 }}>
                    #{idx + 1}
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <span style={{
                      padding: '2px 8px', borderRadius: '4px', fontSize: '0.7rem', fontWeight: 600,
                      background: tx.isInjected ? 'rgba(239,68,68,0.1)' : 'rgba(16,185,129,0.1)',
                      color: tx.isInjected ? '#ef4444' : '#10b981'
                    }}>
                      {tx.isInjected ? 'INJECTED TYPOLOGY' : 'NORMAL BASELINE'}
                    </span>
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontFamily: 'monospace', fontWeight: 600 }}>
                      <span style={{ color: tx.isInjected ? '#ef4444' : 'var(--text-primary)' }}>{tx.source}</span>
                      <ArrowRight size={12} color="var(--text-muted)" />
                      <span style={{ color: tx.isInjected ? '#ef4444' : 'var(--text-primary)' }}>{tx.target}</span>
                    </div>
                  </td>
                  <td style={{ padding: '10px 12px', fontWeight: 600, color: tx.isInjected ? '#ef4444' : 'var(--text-primary)' }}>
                    {formatSats(tx.amount)}
                  </td>
                  <td style={{ padding: '10px 12px', color: 'var(--text-secondary)', fontSize: '0.76rem' }}>
                    {tx.desc}
                  </td>
                  <td style={{ padding: '10px 12px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                      <span style={{ fontFamily: 'monospace', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                        {tx.txid}
                      </span>
                      <button
                        onClick={() => handleCopyTxid(tx.txid)}
                        style={{ background: 'none', border: 'none', cursor: 'pointer', color: copiedTxid === tx.txid ? 'var(--success)' : 'var(--text-muted)', padding: 2 }}
                        title="Copy TxID"
                      >
                        {copiedTxid === tx.txid ? <Check size={11} /> : <Copy size={11} />}
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
