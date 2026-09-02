const API_BASE_URL = 'http://localhost:8000/api/v1';

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
        throw new Error(`Upload failed: ${text}`);
    }
    return response.json();
};

export const pollJobStatus = async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/jobs/${jobId}/status`);
    if (!response.ok) throw new Error('Status check failed');
    return response.json();
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

export const fetchGraphData = async (jobId) => {
    const url = jobId ? `${API_BASE_URL}/jobs/${jobId}/graph/overview` : `${API_BASE_URL}/graph/overview`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch graph data');
    const data = await response.json();
    
    // Map backend JSON to Cytoscape format
    const elements = [];
    (data.nodes || []).forEach(n => {
        const feats = n.wallet_features || {};
        const risk = n.risk_score !== undefined ? n.risk_score : (n.risk_probability || 0);
        const isAnomaly = n.is_anomaly !== undefined ? n.is_anomaly : (risk >= 0.7);
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
    if (!response.ok) throw new Error('Failed to fetch explainability data');
    return response.json();
};

export const fetchTracePattern = async (jobId, walletId) => {
    const url = jobId 
        ? `${API_BASE_URL}/jobs/${jobId}/patterns/trace/${walletId}`
        : `${API_BASE_URL}/patterns/trace/${walletId}`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch trace pattern');
    return response.json();
};

export const fetchAlerts = async (jobId) => {
    const url = jobId ? `${API_BASE_URL}/jobs/${jobId}/alerts` : `${API_BASE_URL}/alerts`;
    const response = await fetch(url);
    if (!response.ok) throw new Error('Failed to fetch alerts');
    return response.json();
};
