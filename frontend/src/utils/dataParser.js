import Papa from 'papaparse';

const loadCSV = async (filePath) => {
  const response = await fetch(filePath);
  const text = await response.text();
  return new Promise((resolve) => {
    Papa.parse(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data)
    });
  });
};

export const loadGraphData = async () => {
  const edgesRaw = await loadCSV('/data/graph_edges.csv');
  const featuresRaw = await loadCSV('/data/graph_features.csv');
  
  // New Final Model Outputs
  const riskRaw = await loadCSV('/data/risk_model_predictions.csv');
  const shapRaw = await loadCSV('/data/top_feature_contributions.csv');

  const elements = [];
  const nodesMap = new Map();

  // Group SHAP features by wallet
  const shapMap = new Map();
  shapRaw.forEach(row => {
      if (!shapMap.has(row.wallet_id)) shapMap.set(row.wallet_id, []);
      shapMap.get(row.wallet_id).push(row);
  });

  featuresRaw.forEach(f => {
    const riskRow = riskRaw.find(s => s.wallet_id === f.wallet_id) || {};
    const shapFeatures = shapMap.get(f.wallet_id) || [];
    
    // Sort SHAP features by rank
    shapFeatures.sort((a, b) => a.rank - b.rank);

    const node = {
      data: {
        id: f.wallet_id,
        label: f.wallet_id,
        degree: f.degree || 0,
        pagerank: f.pagerank || 0,
        community_id: f.community_id,
        // Final Fusion Risk Model properties
        risk_probability: riskRow.risk_probability || 0,
        is_anomaly: riskRow.risk_prediction === 1,
        shap_features: shapFeatures
      }
    };
    nodesMap.set(f.wallet_id, node);
    elements.push(node);
  });

  edgesRaw.forEach(e => {
    if (nodesMap.has(e.source_wallet_id) && nodesMap.has(e.target_wallet_id)) {
      elements.push({
        data: {
          id: `${e.source_wallet_id}-${e.target_wallet_id}`,
          source: e.source_wallet_id,
          target: e.target_wallet_id,
          weight: e.transaction_count || 1,
          amount: e.total_amount_sats || 0
        }
      });
    }
  });

  return elements;
};
