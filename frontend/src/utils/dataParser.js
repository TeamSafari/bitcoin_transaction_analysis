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
  const scoresRaw = await loadCSV('/data/isolation_forest_scores.csv');

  const elements = [];
  const nodesMap = new Map();

  featuresRaw.forEach(f => {
    const scoreRow = scoresRaw.find(s => s.wallet_id === f.wallet_id) || {};
    const node = {
      data: {
        id: f.wallet_id,
        label: f.wallet_id,
        degree: f.degree || 0,
        pagerank: f.pagerank || 0,
        community_id: f.community_id,
        anomaly_score: scoreRow.isolation_forest_anomaly_score || 0,
        is_anomaly: scoreRow.isolation_forest_flag === 1
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
