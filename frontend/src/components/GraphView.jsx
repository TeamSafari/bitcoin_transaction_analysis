import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import cytoscape from 'cytoscape';
import { fetchTracePattern } from '../utils/apiService';

/**
 * Finds the FIRST direct 2-hop cycle involving nodeId: A→B→A.
 * Returns object with the two edge IDs and the intermediate node, or null.
 */
function findDirectCyclePair(cy, nodeId) {
  const outEdges = cy.getElementById(nodeId).outgoers('edge');
  for (let i = 0; i < outEdges.length; i++) {
    const edge = outEdges[i];
    const neighborId = edge.target().id();
    // Check if the neighbor has a direct edge back to nodeId
    const returnEdges = cy.edges(`[source = "${neighborId}"][target = "${nodeId}"]`);
    if (returnEdges.length > 0) {
      return {
        forwardEdge: edge,
        returnEdge: returnEdges[0],
        neighborId,
      };
    }
  }
  return null;
}

const GraphView = forwardRef(({ elements, onNodeSelect, searchQuery, jobId }, ref) => {
  const containerRef = useRef(null);
  const cyRef = useRef(null);

  useEffect(() => {
    if (!containerRef.current || !elements.length) return;

    const cy = cytoscape({
      container: containerRef.current,
      elements: elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': (ele) => ele.data('is_anomaly') ? '#ef4444' : '#3b82f6',
            'width': (ele) => Math.max(16, Math.min(45, (ele.data('degree') || 0) * 0.8 + 12)),
            'height': (ele) => Math.max(16, Math.min(45, (ele.data('degree') || 0) * 0.8 + 12)),
            'label': 'data(id)',
            'color': '#cbd5e1',
            'font-size': '10px',
            'font-family': 'Inter, system-ui, sans-serif',
            'text-valign': 'bottom',
            'text-margin-y': 6,
            'border-width': (ele) => ele.data('is_anomaly') ? 3 : 1.5,
            'border-color': (ele) => ele.data('is_anomaly') ? '#fca5a5' : '#93c5fd',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': (ele) => Math.max(1, Math.min(6, (ele.data('gnn_influence_weight') || 0.5) * 4)),
            'line-color': '#334155',
            'target-arrow-color': '#334155',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'opacity': 0.5
          }
        },
        { selector: '.dimmed', style: { 'display': 'none' } },
        { selector: '.dimmed-soft', style: { 'opacity': 0.1 } },
        { selector: '.highlighted', style: { 'opacity': 1, 'border-width': 4, 'border-color': '#ffffff', 'z-index': 99 } },
        {
          selector: '.incoming-edge',
          style: { 'opacity': 1, 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'width': 4, 'z-index': 10 }
        },
        {
          selector: '.trace-edge',
          style: { 'opacity': 1, 'line-color': '#ef4444', 'target-arrow-color': '#ef4444', 'width': 6, 'z-index': 99 }
        },
        // Cycle edges: bright orange with glow, thick, high z-index
        {
          selector: '.cycle-edge',
          style: {
            'opacity': 1,
            'line-color': '#fb923c',
            'target-arrow-color': '#fb923c',
            'width': 5,
            'z-index': 98,
            'line-style': 'solid',
            'arrow-scale': 1.4,
          }
        },
        // Nodes that form the cycle: ring highlight
        {
          selector: '.cycle-node',
          style: {
            'border-width': 4,
            'border-color': '#fb923c',
            'border-style': 'solid',
            'z-index': 97,
            'opacity': 1,
          }
        },
      ],
      layout: {
        name: 'cose',
        idealEdgeLength: 80,
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 40,
        randomize: false,
        componentSpacing: 100,
        nodeRepulsion: 400000
      }
    });

    cyRef.current = cy;

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      const nodeId = node.id();
      onNodeSelect(node.data());

      // Clear all previous classes
      cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge cycle-edge cycle-node');
      cy.elements().addClass('dimmed-soft');

      // Highlight selected node
      node.removeClass('dimmed-soft').addClass('highlighted');

      // Always show immediate neighbors (yellow incoming, normal outgoing)
      const incomingEdges = node.incomers('edge');
      incomingEdges.removeClass('dimmed-soft').addClass('incoming-edge');
      incomingEdges.sources().removeClass('dimmed-soft');
      node.outgoers('edge').removeClass('dimmed-soft');
      node.outgoers('node').removeClass('dimmed-soft');

      // If there's a direct 2-hop cycle (A→B→A), highlight ONLY those 2 edges
      const cyclePair = findDirectCyclePair(cy, nodeId);
      if (cyclePair) {
        const { forwardEdge, returnEdge, neighborId } = cyclePair;
        // Override the 'incoming-edge' class for these specific edges with 'cycle-edge'
        forwardEdge.removeClass('dimmed-soft incoming-edge').addClass('cycle-edge');
        returnEdge.removeClass('dimmed-soft incoming-edge').addClass('cycle-edge');
        cy.getElementById(neighborId).removeClass('dimmed-soft').addClass('cycle-node');
        // Keep selected node as highlighted (on top)
        node.addClass('highlighted');
      }
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge cycle-edge cycle-node');
        onNodeSelect(null);
      }
    });

    return () => cy.destroy();
  }, [elements]);

  useEffect(() => {
    if (!cyRef.current) return;
    const cy = cyRef.current;

    if (!searchQuery) {
      cy.elements().removeClass('dimmed');
      cy.fit();
      return;
    }

    const node = cy.getElementById(searchQuery);
    if (node.length) {
      const hop1 = node.closedNeighborhood();
      const hop2 = hop1.closedNeighborhood();
      cy.elements().addClass('dimmed');
      hop2.removeClass('dimmed');
      cy.fit(hop2, 50);
      onNodeSelect(node.data());
      cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge cycle-edge cycle-node');
      cy.elements().not('.dimmed').addClass('dimmed-soft');
      node.removeClass('dimmed-soft').addClass('highlighted');

      // Highlight cycle pair for searched node if one exists
      const cyclePair = findDirectCyclePair(cy, searchQuery);
      if (cyclePair) {
        const { forwardEdge, returnEdge, neighborId } = cyclePair;
        if (!forwardEdge.hasClass('dimmed')) {
          forwardEdge.removeClass('dimmed-soft incoming-edge').addClass('cycle-edge');
          returnEdge.removeClass('dimmed-soft incoming-edge').addClass('cycle-edge');
          cy.getElementById(neighborId).removeClass('dimmed-soft').addClass('cycle-node');
        }
        node.addClass('highlighted');
      }

    }
  }, [searchQuery]);

  // Expose intelligent trace function to parent
  useImperativeHandle(ref, () => ({
    traceFunds: async (startNodeId) => {
      if (!cyRef.current || !startNodeId) return;
      const cy = cyRef.current;

      let currentNode = cy.getElementById(startNodeId);
      if (!currentNode.length) return;

      cy.elements().removeClass('incoming-edge trace-edge cycle-edge cycle-node');
      cy.elements().addClass('dimmed-soft');
      currentNode.removeClass('dimmed-soft').addClass('highlighted');

      try {
        // Fetch real ML-detected trace sequence from backend
        const traceData = await fetchTracePattern(jobId, startNodeId);
        const sequence = traceData.sequence || [];

        if (sequence.length > 0) {
          for (let step of sequence) {
            const srcId = step.source_wallet || step.source;
            const tgtId = step.target_wallet || step.target;

            const srcNode = cy.getElementById(srcId);
            const tgtNode = cy.getElementById(tgtId);
            const edge = cy.edges(`[source = "${srcId}"][target = "${tgtId}"]`);

            if (srcNode.length) srcNode.removeClass('dimmed-soft').addClass('highlighted');
            if (tgtNode.length) tgtNode.removeClass('dimmed-soft').addClass('highlighted');

            if (edge.length) {
              edge.removeClass('dimmed-soft').addClass('trace-edge');
              await edge.animate({ style: { 'line-color': '#ef4444', 'width': 6 } }, { duration: 350 }).promise();
              await new Promise(r => setTimeout(r, 150));
            }
          }
          return;
        }
      } catch (err) {
        console.warn('Backend trace pattern fetch failed, falling back to graph traversal:', err);
      }

      // Fallback traversal if backend sequence is empty
      let curr = currentNode;
      for (let i = 0; i < 5; i++) {
        const outEdges = curr.outgoers('edge');
        if (outEdges.length === 0) break;

        let maxEdge = outEdges[0];
        outEdges.forEach(e => {
          if ((e.data('weight') || 0) > (maxEdge.data('weight') || 0)) maxEdge = e;
        });

        maxEdge.removeClass('dimmed-soft').addClass('trace-edge');
        maxEdge.target().removeClass('dimmed-soft').addClass('highlighted');
        await maxEdge.animate({ style: { 'line-color': '#ef4444', 'width': 6 } }, { duration: 350 }).promise();
        await new Promise(r => setTimeout(r, 150));
        curr = maxEdge.target();
      }
    }
  }));

  return <div ref={containerRef} className="cy-wrapper" />;
});

export default GraphView;
