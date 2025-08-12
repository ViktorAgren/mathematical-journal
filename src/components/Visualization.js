import React, { useEffect, useRef } from 'react';

export const Visualization = ({ type, data, title, width = 600, height = 400 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size for retina displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Render based on visualization type
    switch (type) {
      case 'epv-heatmap':
        renderEPVHeatmap(ctx, width, height, data);
        break;
      case 'player-radar':
        renderPlayerRadar(ctx, width, height, data);
        break;
      case 'xg-scatter':
        renderXGScatter(ctx, width, height, data);
        break;
      case 'pitch-control':
        renderPitchControl(ctx, width, height, data);
        break;
      case 'influence-zones':
        renderInfluenceZones(ctx, width, height, data);
        break;
      case 'corner-kick-setup':
        renderCornerKickSetup(ctx, width, height, data);
        break;
      default:
        console.warn(`Unknown visualization type: ${type}`);
    }
  }, [type, data, width, height]);

  return (
    <div className="visualization-container">
      {title && <h4 className="visualization-title">{title}</h4>}
      <div className="visualization-canvas-wrapper">
        <canvas
          ref={canvasRef}
          style={{ width: `${width}px`, height: `${height}px` }}
          className="visualization-canvas"
        />
      </div>
    </div>
  );
};

// EPV Heatmap for Soccer Position Models
const renderEPVHeatmap = (ctx, width, height, data) => {
  const pitchWidth = width - 80;
  const pitchHeight = height - 80;
  const offsetX = 40;
  const offsetY = 40;
  
  // Draw pitch outline
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.strokeRect(offsetX, offsetY, pitchWidth, pitchHeight);
  
  // Draw center line
  ctx.beginPath();
  ctx.moveTo(offsetX + pitchWidth / 2, offsetY);
  ctx.lineTo(offsetX + pitchWidth / 2, offsetY + pitchHeight);
  ctx.stroke();
  
  // Draw penalty areas
  const penaltyWidth = pitchWidth * 0.15;
  const penaltyHeight = pitchHeight * 0.4;
  const penaltyY = offsetY + (pitchHeight - penaltyHeight) / 2;
  
  ctx.strokeRect(offsetX, penaltyY, penaltyWidth, penaltyHeight);
  ctx.strokeRect(offsetX + pitchWidth - penaltyWidth, penaltyY, penaltyWidth, penaltyHeight);
  
  // Create EPV heatmap zones
  const zones = [
    { x: 0, y: 0, w: 0.3, h: 1, value: 0.05, label: 'Defensive Third' },
    { x: 0.3, y: 0, w: 0.4, h: 1, value: 0.15, label: 'Middle Third' },
    { x: 0.7, y: 0.2, w: 0.3, h: 0.6, value: 0.35, label: 'Final Third' },
    { x: 0.85, y: 0.3, w: 0.15, h: 0.4, value: 0.65, label: 'Penalty Area' },
  ];
  
  zones.forEach(zone => {
    const x = offsetX + zone.x * pitchWidth;
    const y = offsetY + zone.y * pitchHeight;
    const w = zone.w * pitchWidth;
    const h = zone.h * pitchHeight;
    
    // Color based on EPV value (red = high threat, blue = low threat)
    const intensity = zone.value;
    const red = Math.floor(255 * intensity);
    const blue = Math.floor(255 * (1 - intensity));
    
    ctx.fillStyle = `rgba(${red}, 100, ${blue}, 0.4)`;
    ctx.fillRect(x, y, w, h);
    
    // Add EPV value text
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(zone.value.toFixed(2), x + w/2, y + h/2 - 5);
    ctx.font = '10px Arial';
    ctx.fillText(zone.label, x + w/2, y + h/2 + 10);
  });
  
  // Add title
  ctx.fillStyle = '#333333';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Expected Possession Value by Pitch Zone', width/2, 25);
};

// Player Radar Chart
const renderPlayerRadar = (ctx, width, height, data) => {
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 3;
  
  const attributes = data.attributes || [
    'Finishing', 'Positioning', 'Passing', 'Dribbling', 'Defense', 'Physical'
  ];
  const playerData = data.values || [0.85, 0.78, 0.92, 0.73, 0.45, 0.88];
  
  const angleStep = (2 * Math.PI) / attributes.length;
  
  // Draw concentric circles
  for (let i = 1; i <= 5; i++) {
    ctx.beginPath();
    ctx.arc(centerX, centerY, (radius * i) / 5, 0, 2 * Math.PI);
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  
  // Draw axes and labels
  attributes.forEach((attr, index) => {
    const angle = index * angleStep - Math.PI / 2;
    const x = centerX + Math.cos(angle) * radius;
    const y = centerY + Math.sin(angle) * radius;
    
    // Draw axis line
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.strokeStyle = '#e0e0e0';
    ctx.stroke();
    
    // Draw label
    ctx.fillStyle = '#333333';
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    const labelX = centerX + Math.cos(angle) * (radius + 20);
    const labelY = centerY + Math.sin(angle) * (radius + 20);
    ctx.fillText(attr, labelX, labelY);
  });
  
  // Draw player data polygon
  ctx.beginPath();
  playerData.forEach((value, index) => {
    const angle = index * angleStep - Math.PI / 2;
    const x = centerX + Math.cos(angle) * (radius * value);
    const y = centerY + Math.sin(angle) * (radius * value);
    
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.closePath();
  ctx.fillStyle = 'rgba(59, 130, 246, 0.3)';
  ctx.fill();
  ctx.strokeStyle = '#3b82f6';
  ctx.lineWidth = 2;
  ctx.stroke();
  
  // Draw data points
  playerData.forEach((value, index) => {
    const angle = index * angleStep - Math.PI / 2;
    const x = centerX + Math.cos(angle) * (radius * value);
    const y = centerY + Math.sin(angle) * (radius * value);
    
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = '#3b82f6';
    ctx.fill();
  });
  
  // Add title
  ctx.fillStyle = '#333333';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText(data.playerName || 'Player Performance Profile', width/2, 25);
};

// Expected Goals Scatter Plot
const renderXGScatter = (ctx, width, height, data) => {
  const margin = 60;
  const plotWidth = width - 2 * margin;
  const plotHeight = height - 2 * margin;
  
  // Draw axes
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(margin, height - margin);
  ctx.lineTo(width - margin, height - margin);
  ctx.moveTo(margin, height - margin);
  ctx.lineTo(margin, margin);
  ctx.stroke();
  
  // Sample shot data
  const shots = data.shots || [
    { distance: 8, angle: 45, xG: 0.8, goal: true },
    { distance: 12, angle: 30, xG: 0.6, goal: true },
    { distance: 18, angle: 15, xG: 0.25, goal: false },
    { distance: 25, angle: 60, xG: 0.1, goal: false },
    { distance: 6, angle: 90, xG: 0.9, goal: true },
    { distance: 20, angle: 20, xG: 0.3, goal: false },
    { distance: 15, angle: 40, xG: 0.4, goal: true },
  ];
  
  // Plot shots
  shots.forEach(shot => {
    const x = margin + (shot.distance / 30) * plotWidth;
    const y = height - margin - (shot.angle / 90) * plotHeight;
    const radius = 3 + shot.xG * 8;
    
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, 2 * Math.PI);
    ctx.fillStyle = shot.goal ? 'rgba(34, 197, 94, 0.7)' : 'rgba(239, 68, 68, 0.7)';
    ctx.fill();
    ctx.strokeStyle = shot.goal ? '#22c55e' : '#ef4444';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  
  // Add labels
  ctx.fillStyle = '#333333';
  ctx.font = '12px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Shot Distance (yards)', width/2, height - 10);
  
  ctx.save();
  ctx.translate(15, height/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText('Shot Angle (degrees)', 0, 0);
  ctx.restore();
  
  // Add legend
  ctx.font = '10px Arial';
  ctx.fillStyle = '#22c55e';
  ctx.fillText('● Goal', width - 100, 40);
  ctx.fillStyle = '#ef4444';
  ctx.fillText('● Miss', width - 100, 55);
  ctx.fillStyle = '#333333';
  ctx.fillText('Size = xG', width - 100, 70);
  
  // Add title
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Shot Quality vs Distance & Angle', width/2, 25);
};

// Pitch Control Heatmap
const renderPitchControl = (ctx, width, height, data) => {
  const pitchWidth = width - 80;
  const pitchHeight = height - 80;
  const offsetX = 40;
  const offsetY = 40;
  
  // Draw pitch
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.strokeRect(offsetX, offsetY, pitchWidth, pitchHeight);
  
  // Draw center line and circle
  ctx.beginPath();
  ctx.moveTo(offsetX + pitchWidth / 2, offsetY);
  ctx.lineTo(offsetX + pitchWidth / 2, offsetY + pitchHeight);
  ctx.stroke();
  
  ctx.beginPath();
  ctx.arc(offsetX + pitchWidth / 2, offsetY + pitchHeight / 2, 40, 0, 2 * Math.PI);
  ctx.stroke();
  
  // Create grid for control visualization
  const gridSize = 20;
  const rows = Math.floor(pitchHeight / gridSize);
  const cols = Math.floor(pitchWidth / gridSize);
  
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      const x = offsetX + j * gridSize;
      const y = offsetY + i * gridSize;
      
      // Simulate control values (would come from actual model)
      const centerX = j / cols;
      const centerY = i / rows;
      const distanceFromCenter = Math.sqrt((centerX - 0.5) ** 2 + (centerY - 0.5) ** 2);
      const control = Math.sin(centerX * Math.PI) * Math.cos(centerY * Math.PI) * (1 - distanceFromCenter);
      
      // Color based on control (-1 = team B, +1 = team A)
      const intensity = Math.abs(control);
      const red = control > 0 ? Math.floor(255 * intensity) : 0;
      const blue = control < 0 ? Math.floor(255 * intensity) : 0;
      
      ctx.fillStyle = `rgba(${red}, 50, ${blue}, ${intensity * 0.6})`;
      ctx.fillRect(x, y, gridSize, gridSize);
    }
  }
  
  // Add player positions
  const players = data.players || [
    { x: 0.2, y: 0.5, team: 'A' },
    { x: 0.4, y: 0.3, team: 'A' },
    { x: 0.6, y: 0.7, team: 'A' },
    { x: 0.8, y: 0.4, team: 'B' },
    { x: 0.7, y: 0.6, team: 'B' },
  ];
  
  players.forEach(player => {
    const x = offsetX + player.x * pitchWidth;
    const y = offsetY + player.y * pitchHeight;
    
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = player.team === 'A' ? '#ef4444' : '#3b82f6';
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
  });
  
  // Add title
  ctx.fillStyle = '#333333';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Pitch Control Map', width/2, 25);
  
  // Add legend
  ctx.font = '10px Arial';
  ctx.fillStyle = '#ef4444';
  ctx.fillText('● Team A', 20, height - 40);
  ctx.fillStyle = '#3b82f6';
  ctx.fillText('● Team B', 20, height - 25);
  ctx.fillStyle = '#333333';
  ctx.fillText('Red = Team A Control, Blue = Team B Control', 20, height - 10);
};

// Influence Zones (simplified version)
const renderInfluenceZones = (ctx, width, height, data) => {
  renderPitchControl(ctx, width, height, data); // Reuse pitch control for now
};

// Corner Kick Setup
const renderCornerKickSetup = (ctx, width, height, data) => {
  const pitchWidth = width - 80;
  const pitchHeight = height - 80;
  const offsetX = 40;
  const offsetY = 40;
  
  // Draw partial pitch (penalty area focus)
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  
  // Goal line
  ctx.beginPath();
  ctx.moveTo(offsetX + pitchWidth * 0.8, offsetY);
  ctx.lineTo(offsetX + pitchWidth * 0.8, offsetY + pitchHeight);
  ctx.stroke();
  
  // Penalty area
  const penaltyWidth = pitchWidth * 0.2;
  const penaltyHeight = pitchHeight * 0.6;
  const penaltyY = offsetY + (pitchHeight - penaltyHeight) / 2;
  
  ctx.strokeRect(offsetX + pitchWidth * 0.8 - penaltyWidth, penaltyY, penaltyWidth, penaltyHeight);
  
  // Goal
  ctx.strokeRect(offsetX + pitchWidth * 0.8, offsetY + pitchHeight * 0.35, 20, pitchHeight * 0.3);
  
  // Corner arc
  ctx.beginPath();
  ctx.arc(offsetX + pitchWidth * 0.8, offsetY + pitchHeight, 30, -Math.PI/2, 0);
  ctx.stroke();
  
  // Ball position
  ctx.beginPath();
  ctx.arc(offsetX + pitchWidth * 0.8 - 5, offsetY + pitchHeight - 5, 5, 0, 2 * Math.PI);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  ctx.strokeStyle = '#000000';
  ctx.stroke();
  
  // Player positions for corner kick
  const cornerPlayers = [
    { x: 0.75, y: 0.4, team: 'attacking', role: 'Near post' },
    { x: 0.7, y: 0.5, team: 'attacking', role: 'Central' },
    { x: 0.65, y: 0.6, team: 'attacking', role: 'Far post' },
    { x: 0.6, y: 0.45, team: 'attacking', role: 'Edge of box' },
    { x: 0.78, y: 0.42, team: 'defending', role: 'Marking' },
    { x: 0.72, y: 0.52, team: 'defending', role: 'Marking' },
    { x: 0.67, y: 0.62, team: 'defending', role: 'Marking' },
    { x: 0.85, y: 0.5, team: 'defending', role: 'Goalkeeper' },
  ];
  
  cornerPlayers.forEach(player => {
    const x = offsetX + player.x * pitchWidth;
    const y = offsetY + player.y * pitchHeight;
    
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, 2 * Math.PI);
    
    if (player.team === 'attacking') {
      ctx.fillStyle = '#ef4444';
    } else if (player.role === 'Goalkeeper') {
      ctx.fillStyle = '#22c55e';
    } else {
      ctx.fillStyle = '#3b82f6';
    }
    
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Add role label
    ctx.fillStyle = '#333333';
    ctx.font = '8px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(player.role, x, y - 12);
  });
  
  // Add movement arrows (simplified)
  const movements = [
    { from: { x: 0.5, y: 0.9 }, to: { x: 0.75, y: 0.4 } }, // Corner taker to near post
    { from: { x: 0.65, y: 0.3 }, to: { x: 0.7, y: 0.5 } }, // Player making run
  ];
  
  movements.forEach(movement => {
    const fromX = offsetX + movement.from.x * pitchWidth;
    const fromY = offsetY + movement.from.y * pitchHeight;
    const toX = offsetX + movement.to.x * pitchWidth;
    const toY = offsetY + movement.to.y * pitchHeight;
    
    ctx.beginPath();
    ctx.moveTo(fromX, fromY);
    ctx.lineTo(toX, toY);
    ctx.strokeStyle = '#fbbf24';
    ctx.lineWidth = 3;
    ctx.stroke();
    
    // Arrow head
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const headLength = 10;
    ctx.beginPath();
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLength * Math.cos(angle - Math.PI/6), toY - headLength * Math.sin(angle - Math.PI/6));
    ctx.moveTo(toX, toY);
    ctx.lineTo(toX - headLength * Math.cos(angle + Math.PI/6), toY - headLength * Math.sin(angle + Math.PI/6));
    ctx.stroke();
  });
  
  // Add title
  ctx.fillStyle = '#333333';
  ctx.font = 'bold 14px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Optimized Corner Kick Setup', width/2, 25);
  
  // Add legend
  ctx.font = '10px Arial';
  ctx.fillStyle = '#ef4444';
  ctx.fillText('● Attacking Team', 20, height - 55);
  ctx.fillStyle = '#3b82f6';
  ctx.fillText('● Defending Team', 20, height - 40);
  ctx.fillStyle = '#22c55e';
  ctx.fillText('● Goalkeeper', 20, height - 25);
  ctx.fillStyle = '#fbbf24';
  ctx.fillText('→ Player Movement', 20, height - 10);
};