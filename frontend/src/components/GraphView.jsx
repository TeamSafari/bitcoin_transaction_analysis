import React, { useEffect, useRef } from 'react';
import cytoscape from 'cytoscape';

export default function GraphView({ elements, onNodeSelect, searchQuery }) {
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
            'width': (ele) => Math.max(15, (ele.data('degree') || 0) * 1.5),
            'height': (ele) => Math.max(15, (ele.data('degree') || 0) * 1.5),
            'label': 'data(id)',
            'color': '#94a3b8',
            'font-size': '10px',
            'text-valign': 'bottom',
            'text-margin-y': 5,
            'border-width': 2,
            'border-color': (ele) => ele.data('is_anomaly') ? '#fca5a5' : '#93c5fd',
          }
        },
        {
          selector: 'edge',
          style: {
            'width': (ele) => Math.max(1, Math.log10(ele.data('weight') || 10)),
            'line-color': '#334155',
            'target-arrow-color': '#334155',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'opacity': 0.6
          }
        },
        { selector: '.dimmed', style: { 'display': 'none' } }, 
        { selector: '.dimmed-soft', style: { 'opacity': 0.1 } }, 
        { selector: '.highlighted', style: { 'opacity': 1, 'border-width': 4, 'border-color': '#fff' } },
        { selector: '.incoming-edge', style: { 'opacity': 1, 'line-color': '#f59e0b', 'target-arrow-color': '#f59e0b', 'width': 3, 'z-index': 10 } }
      ],
      layout: { name: 'cose', idealEdgeLength: 100, nodeOverlap: 20, refresh: 20, fit: true, padding: 30, randomize: false, componentSpacing: 100 }
    });

    cyRef.current = cy;

    cy.on('tap', 'node', (evt) => {
      const node = evt.target;
      onNodeSelect(node.data());
      cy.elements().removeClass('dimmed-soft highlighted incoming-edge');
      cy.elements().addClass('dimmed-soft');
      node.removeClass('dimmed-soft').addClass('highlighted');
      const incomingEdges = node.incomers('edge');
      incomingEdges.removeClass('dimmed-soft').addClass('incoming-edge');
      incomingEdges.sources().removeClass('dimmed-soft');
    });

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        cy.elements().removeClass('dimmed-soft highlighted incoming-edge');
        onNodeSelect(null);
      }
    });

    return () => cy.destroy();
  }, [elements]);

  // Contextual Sub-Graph Logic
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
        // Find 2-hop neighborhood
        const hop1 = node.closedNeighborhood();
        const hop2 = hop1.closedNeighborhood();
        
        cy.elements().addClass('dimmed');
        hop2.removeClass('dimmed');
        cy.fit(hop2, 50);
        
        // Select it
        onNodeSelect(node.data());
        cy.elements().removeClass('dimmed-soft highlighted incoming-edge');
        cy.elements().not('.dimmed').addClass('dimmed-soft');
        node.removeClass('dimmed-soft').addClass('highlighted');
    }
  }, [searchQuery]);

  window.traceFunds = async () => {
    if (!cyRef.current) return;
    const cy = cyRef.current;
    
    // Mocking traceability
    const edges = cy.edges().toArray().slice(0, 5); 
    cy.elements().addClass('dimmed-soft');
    
    for (let edge of edges) {
      edge.removeClass('dimmed-soft').addClass('incoming-edge');
      edge.source().removeClass('dimmed-soft').addClass('highlighted');
      edge.target().removeClass('dimmed-soft').addClass('highlighted');
      await edge.animate({ style: { 'line-color': '#ef4444', 'width': 6 } }, { duration: 400 }).promise();
      await new Promise(r => setTimeout(r, 200));
    }
  };

  return <div ref={containerRef} className="cy-wrapper" />;
}
