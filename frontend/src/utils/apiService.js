const API_BASE_URL = 'http://localhost:8000/api/v1';

const handleResponse = async (response) => {
    if (response.status === 202) {
        // Job still processing — return the body as-is (it's a valid detail message)
        const data = await response.json().catch(() => ({}));
        return { status: 'processing', progress: data.detail ? 50 : 0, step: String(data.detail || 'Processing...') };
    }
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text);
    }
    return response.json();
};

export const uploadCSV = async (fileMap) => {
    const formData = new FormData();
    for (const [key, file] of Object.entries(fileMap)) {
        formData.append(key, file);
    }
    const response = await fetch(`${API_BASE_URL}/jobs/upload`, {
        method: 'POST',
        body: formData
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`Upload failed (${response.status}): ${text}`);
    }
    return response.json();
};

export const pollJobStatus = async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`);
    return handleResponse(response);
};

export const fetchJobsList = async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/jobs`);
        if (!response.ok) return [];
        return await response.json();
    } catch (e) {
        console.warn('Could not fetch jobs list:', e);
        return [];
    }
};

export const fetchJobSummary = async (jobId) => {
    try {
        const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/summary`);
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.warn('Could not fetch job summary:', e);
        return null;
    }
};

export const fetchGraphData = async (jobId, retries = 20) => {
    const url = jobId
        ? `${API_BASE_URL}/jobs/${jobId}/graph/overview?max_nodes=500`
        : `${API_BASE_URL}/graph/overview?max_nodes=500`;
    
    let response;
    for (let i = 0; i < retries; i++) {
        response = await fetch(url);
        if (response.status !== 202) break;
        await new Promise(r => setTimeout(r, 1500));
    }
    
    if (response.status === 202) throw new Error('Job is still processing after retries');
    if (!response.ok) throw new Error(`Failed to fetch graph data (${response.status})`);
    
    const data = await response.json();
    
    // Map backend JSON to Cytoscape format
    // IMPORTANT: trust backend's is_anomaly boolean — do NOT re-derive from risk_probability
    const elements = [];
    (data.nodes || []).forEach(n => {
        const feats = n.wallet_features || {};
        const risk = n.risk_score !== undefined ? n.risk_score : (n.risk_probability || 0);

        // Trust the backend's classification flag directly
        const isAnomaly = n.is_anomaly === true;

        const degree = n.degree || feats.degree || ((feats.in_degree || 0) + (feats.out_degree || 0)) || 0;
        const pagerank = n.pagerank || feats.pagerank || 0;
        const community = n.community_id || feats.community_id || '0';

        elements.push({
            data: {
                id: n.id || n.wallet_id,
                label: n.id || n.wallet_id,
                degree: degree,
                in_degree: feats.in_degree || 0,
                out_degree: feats.out_degree || 0,
                visible_degree: feats.visible_degree ?? degree,
                visible_in_degree: feats.visible_in_degree ?? (feats.in_degree || 0),
                visible_out_degree: feats.visible_out_degree ?? (feats.out_degree || 0),
                pagerank: pagerank,
                community_id: community,
                betweenness_centrality: feats.betweenness_centrality || 0,
                clustering_coefficient: feats.clustering_coefficient || 0,
                risk_probability: risk,
                is_anomaly: isAnomaly,
                tx_count: feats.tx_count || 0,
                total_sent_sats: feats.total_sent_sats || 0,
                total_received_sats: feats.total_received_sats || 0,
                wallet_features: feats,
            }
        });
    });
    
    (data.edges || []).forEach(e => {
        elements.push({
            data: {
                id: `${e.source}-${e.target}`,
                source: e.source,
                target: e.target,
                weight: e.transaction_count || 1,
                total_amount_sats: e.total_amount_sats || 0,
                gnn_influence_weight: e.gnn_influence_weight || 0.5,
                frequency_score: e.frequency_score || 0.5,
            }
        });
    });
    
    return elements;
};

export const fetchExplainability = async (jobId, walletId) => {
    const url = jobId 
        ? `${API_BASE_URL}/jobs/${jobId}/explainability/wallet/${walletId}`
        : `${API_BASE_URL}/explainability/wallet/${walletId}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch explainability data (${response.status})`);
    return response.json();
};

export const fetchTracePattern = async (jobId, walletId) => {
    const url = jobId 
        ? `${API_BASE_URL}/jobs/${jobId}/patterns/trace/${walletId}`
        : `${API_BASE_URL}/patterns/trace/${walletId}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch trace pattern (${response.status})`);
    return response.json();
};

export const fetchAlerts = async (jobId) => {
    const url = jobId ? `${API_BASE_URL}/jobs/${jobId}/alerts` : `${API_BASE_URL}/alerts`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch alerts (${response.status})`);
    return response.json();
};

export const fetchLLMExplainability = async (jobId, walletId) => {
    const url = jobId 
        ? `${API_BASE_URL}/jobs/${jobId}/explain/wallet/${walletId}`
        : `${API_BASE_URL}/explain/wallet/${walletId}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch LLM explainability data (${response.status})`);
    return response.json();
};
