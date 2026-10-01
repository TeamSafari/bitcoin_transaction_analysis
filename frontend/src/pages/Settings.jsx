import React from 'react';
import { Settings as SettingsIcon } from 'lucide-react';

export default function Settings() {
  return (
    <div style={{ padding: '32px', height: '100%', overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
        <SettingsIcon size={28} color="var(--text-secondary)" />
        <h1 style={{ fontSize: '1.5rem', fontWeight: 600 }}>System Settings</h1>
      </div>
      
      <div className="data-card" style={{ maxWidth: '600px', padding: '24px' }}>
         <h3 style={{ marginBottom: '16px' }}>Detection Thresholds</h3>
         <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)' }}>Risk Score Threshold (Anomaly Flag)</label>
            <input type="range" min="0" max="100" defaultValue="60" style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
               <span>0</span>
               <span>0.60 (60%)</span>
               <span>100</span>
            </div>
         </div>
         
         <hr style={{ border: 'none', borderTop: '1px solid var(--border)', margin: '24px 0' }} />
         
         <h3 style={{ marginBottom: '16px' }}>LLM Configuration</h3>
         <div style={{ marginBottom: '16px' }}>
            <label style={{ display: 'block', marginBottom: '8px', color: 'var(--text-secondary)' }}>Local Model Path</label>
            <input type="text" defaultValue="models/artifacts/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf" style={{ width: '100%', background: 'var(--bg-panel)', border: '1px solid var(--border)', padding: '10px', color: 'var(--text-primary)', borderRadius: '6px' }} />
         </div>
      </div>
    </div>
  );
}
