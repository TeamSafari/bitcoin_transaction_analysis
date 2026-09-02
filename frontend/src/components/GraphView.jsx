import React, { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import cytoscape from 'cytoscape';
import { fetchTracePattern } from '../utils/apiService';

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
        { selector: '.incoming-edge', style: { 'opacity': 1, 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'width': 4, 'z-index': 10 } },
        { selector: '.trace-edge', style: { 'opacity': 1, 'line-color': '#ef4444', 'target-arrow-color': '#ef4444', 'width': 6, 'z-index': 99 } }
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
      onNodeSelect(node.data());
      cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge');
      cy.elements().addClass('dimmed-soft');
      node.removeClass('dimmed-soft').addClass('highlighted');
      const incomingEdges = node.incomers('edge');
      incomingEdges.removeClass('dimmed-soft').addClass('incoming-edge');
      incomingEdges.sources().removeClass('dimmed-soft');
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge');
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
        cy.elements().removeClass('dimmed-soft highlighted incoming-edge trace-edge');
        cy.elements().not('.dimmed').addClass('dimmed-soft');
        node.removeClass('dimmed-soft').addClass('highlighted');
    }
  }, [searchQuery]);

  // Expose intelligent trace function to parent
  useImperativeHandle(ref, () => ({
    traceFunds: async (startNodeId) => {
      if (!cyRef.current || !startNodeId) return;
      const cy = cyRef.current;
      
      let currentNode = cy.getElementById(startNodeId);
      if (!currentNode.length) return;

      cy.elements().removeClass('incoming-edge trace-edge');
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
