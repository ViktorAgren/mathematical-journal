import React from 'react';

export const Navigation = ({ onBrandClick, onBackToList, showBackButton, selectedPaper }) => {
  return (
    <nav className="journal-nav">
      <div className="nav-content">
        <div className="nav-left">
          <button 
            onClick={onBrandClick}
            className="nav-brand"
          >
            <div className="brand-title">Insights</div>
          </button>
        </div>
        
        <div className="nav-center">
          {selectedPaper && (
            <div className="nav-paper-info">
              <h1 className="nav-paper-title">{selectedPaper.title}</h1>
              <p className="nav-paper-subtitle">{selectedPaper.subtitle}</p>
            </div>
          )}
        </div>
        
        <div className="nav-right">
          {showBackButton && (
            <button 
              onClick={onBackToList}
              className="nav-back-btn"
            >
              <svg className="nav-back-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              All Papers
            </button>
          )}
          
          <div className="nav-actions">
            <button className="nav-action-btn" title="Share">
              <svg className="nav-action-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.367 2.684 3 3 0 00-5.367-2.684z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </nav>
  );
};