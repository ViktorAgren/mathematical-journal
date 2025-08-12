import React from 'react';

export const Theorem = ({ title, children }) => {
  return (
    <div className="theorem-block">
      <div className="theorem-header">
        <div className="theorem-label">Theorem</div>
        {title && <div className="theorem-title">{title}</div>}
      </div>
      <div className="theorem-content">
        {children}
      </div>
    </div>
  );
};