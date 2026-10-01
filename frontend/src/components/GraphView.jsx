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

const GraphView = forwardRef(({ elements, onNodeSelect, searchQuery, jobId, isDarkMode }, ref) => {
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
            // 🔴 Anomaly = red, 🟢 Normal = green
            'background-color': (ele) => ele.data('is_anomaly') ? '#e85454' : '#3dd68c',
            'width': (ele) => Math.max(8, Math.min(20, (ele.data('degree') || 0) * 0.3 + 8)),
            'height': (ele) => Math.max(8, Math.min(20, (ele.data('degree') || 0) * 0.3 + 8)),
            'label': 'data(id)',
            'color': isDarkMode ? '#cbd5e1' : '#64748b',
            'font-size': '6px',
            'font-family': 'Inter, system-ui, sans-serif',
            'text-valign': 'bottom',
            'text-margin-y': 3,
            'border-width': 1,
            'border-color': (ele) => ele.data('is_anomaly') ? 'rgba(232,84,84,0.6)' : 'rgba(61,214,140,0.5)',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': 0.8,
            'line-color': isDarkMode ? 'rgba(71,85,105,0.4)' : 'rgba(100,116,139,0.35)',
            'target-arrow-color': isDarkMode ? 'rgba(71,85,105,0.4)' : 'rgba(100,116,139,0.35)',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.5,
            'curve-style': 'bezier',
            'opacity': isDarkMode ? 0.35 : 0.6
          }
        },
        { selector: '.dimmed', style: { 'display': 'none' } },
        { selector: '.dimmed-soft', style: { 'opacity': 0.06 } },
        {
          selector: '.highlighted',
          style: { 'opacity': 1, 'border-width': 2, 'border-color': isDarkMode ? '#e2e8f0' : '#1e293b', 'z-index': 99 }
        },
        {
          selector: '.incoming-edge',
          style: { 'opacity': 1, 'line-color': '#f0a142', 'target-arrow-color': '#f0a142', 'width': 1.8, 'z-index': 10, 'arrow-scale': 1.8 }
        },
        {
          selector: '.outgoing-edge',
          style: { 'opacity': 0.55, 'line-color': isDarkMode ? 'rgba(71,85,105,0.6)' : 'rgba(100,116,139,0.5)', 'width': 1.0, 'z-index': 5, 'arrow-scale': 1.5 }
        },
        {
          selector: '.trace-edge',
          style: { 'opacity': 1, 'line-color': '#e85454', 'target-arrow-color': '#e85454', 'width': 2.5, 'z-index': 99, 'arrow-scale': 2.0 }
        },
        {
          selector: '.cycle-edge',
          style: {
            'opacity': 1, 'line-color': '#fb923c', 'target-arrow-color': '#fb923c',
            'width': 2.5, 'z-index': 98, 'arrow-scale': 2.0,
          }
        },
        {
          selector: '.cycle-node',
          style: { 'border-width': 2, 'border-color': '#fb923c', 'z-index': 97, 'opacity': 1 }
        },
      ],
      layout: {
        name: 'cose',
        idealEdgeLength: 400,       // Further increased to stretch connections
        nodeOverlap: 20,
        refresh: 20,
        fit: true,
        padding: 50,
        randomize: true,
        componentSpacing: 400,
        nodeRepulsion: 8000000,     // Doubled repulsion
        edgeElasticity: 30,         // Reduced elasticity so edges can stretch more
        nestingFactor: 5,
        gravity: 0.05,              // Halved gravity to prevent it from pulling into a tight circle
        numIter: 1500,
        initialTemp: 300,
        coolingFactor: 0.98,
        minTemp: 1.0
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

    return () => {
      if (cy) {
        cy.stop();        // Stop any running layout animation
        cyRef.current = null;  // Clear ref BEFORE destroy to prevent stale access
        cy.destroy();
      }
    };
  }, [elements, isDarkMode]);

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
