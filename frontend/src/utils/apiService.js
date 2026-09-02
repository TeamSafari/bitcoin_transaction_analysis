import { loadGraphData as loadMockGraphData } from './dataParser';

// Default to localhost:8000 for FastAPI/Flask. Change this to match your backend!
const API_BASE_URL = 'http://localhost:8000/api/v1';

// MOCK FLAG: Set to false once your backend is actually running.
// If true, it simulates the backend API calls but returns the local CSV data.
const USE_MOCK_API = false; 

export const uploadCSV = async (file) => {
    if (USE_MOCK_API) {
        return new Promise(resolve => setTimeout(() => resolve({ job_id: 'mock-123', status: 'processing' }), 1000));
    }

    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/jobs/upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) throw new Error('Upload failed');
    return response.json();
};

export const pollJobStatus = async (jobId) => {
    if (USE_MOCK_API) {
        // Simulate a processing sequence
        return new Promise(resolve => {
            const steps = [
                { progress: 25, step: "Feature Engineering..." },
                { progress: 50, step: "Running GraphSAGE..." },
                { progress: 75, step: "Calculating SHAP values..." },
                { progress: 100, step: "Complete" }
            ];
            // Just return a random step for demo purposes
            const randomStep = steps[Math.floor(Math.random() * steps.length)];
            resolve(randomStep);
        });
    }

    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`);
    if (!response.ok) throw new Error('Status check failed');
    return response.json();
};

export const fetchGraphData = async (jobId) => {
    if (USE_MOCK_API) {
        return await loadMockGraphData(); // Fallback to our existing CSV parser
    }

    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/graph/overview`);
    if (!response.ok) throw new Error('Failed to fetch graph data');
    const data = await response.json();
    
    // Map backend JSON to Cytoscape format
    const elements = [];
    data.nodes.forEach(n => {
        elements.push({
            data: {
                id: n.wallet_id || n.id,
                label: n.wallet_id || n.id,
                degree: n.degree || 0,
                pagerank: n.pagerank || 0,
                community_id: n.community_id,
                risk_probability: n.risk_probability || 0,
                is_anomaly: n.is_anomaly || (n.risk_prediction === 1)
            }
        });
    });
    
    data.edges.forEach(e => {
        elements.push({
            data: {
                id: `${e.source}-${e.target}`,
                source: e.source,
                target: e.target,
                weight: e.transaction_count || 1,
                gnn_influence_weight: e.gnn_influence_weight || 0
            }
        });
    });
    
    return elements;
};

export const fetchExplainability = async (jobId, walletId) => {
    if (USE_MOCK_API) {
        // Mock fallback for SHAP: just returning a dummy array if the real API isn't up
        return {
            top_risk_factors: [
                { feature: "median_sent_sats", feature_value: 17.17, shap_value: 2.4331, direction: "increases_risk" },
                { feature: "isolation_forest_score", feature_value: 0.12, shap_value: 0.78, direction: "increases_risk" }
            ]
        };
    }

    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/explainability/wallet/${walletId}`);
    if (!response.ok) throw new Error('Failed to fetch explainability data');
    return response.json();
};
