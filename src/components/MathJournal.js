import React, { useState } from 'react';
import { PaperList } from './PaperList';
import { Article } from './Article';
import { Navigation } from './Navigation';
import { getPaper } from '../content/papers';

export const MathJournal = () => {
  const [selectedPaperId, setSelectedPaperId] = useState(null);
  const [showPaperList, setShowPaperList] = useState(true);

  const selectedPaper = selectedPaperId ? getPaper(selectedPaperId) : null;

  const handleSelectPaper = (paperId) => {
    setSelectedPaperId(paperId);
    setShowPaperList(false);
  };

  const handleBackToList = () => {
    setShowPaperList(true);
    setSelectedPaperId(null);
  };

  const handleBrandClick = () => {
    setShowPaperList(true);
    setSelectedPaperId(null);
  };

  return (
    <div className="math-journal">
      <Navigation 
        onBrandClick={handleBrandClick}
        onBackToList={handleBackToList}
        showBackButton={!showPaperList && selectedPaper}
        selectedPaper={selectedPaper}
      />
      
      <main className="journal-main">
        {showPaperList ? (
          <PaperList onSelectPaper={handleSelectPaper} />
        ) : selectedPaper ? (
          <Article paper={selectedPaper} />
        ) : (
          <PaperList onSelectPaper={handleSelectPaper} />
        )}
      </main>
      
      <footer className="journal-footer">
        <div className="footer-content">
          <p className="footer-author">
            Viktor Ågren • {new Date().getFullYear()}
          </p>
        </div>
      </footer>
    </div>
  );
};