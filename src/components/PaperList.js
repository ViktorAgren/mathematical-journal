import React from "react";
import { getAllPapers } from "../content/papers";

export const PaperList = ({ onSelectPaper }) => {
  const papers = getAllPapers();

  return (
    <div className="paper-list">
      <div className="paper-list-header">
      </div>

      <div className="papers-grid">
        {papers.map((paper) => (
          <div
            key={paper.id}
            className="paper-card"
            onClick={() => onSelectPaper(paper.id)}
          >
            <div className="paper-card-header">
              <div className="paper-card-tags">
                {paper.tags.map((tag) => (
                  <span key={tag} className="paper-card-tag">
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <h3 className="paper-card-title">{paper.title}</h3>
            <p className="paper-card-subtitle">{paper.subtitle}</p>

            <p className="paper-card-abstract">{paper.abstract}</p>

            <div className="paper-card-footer">
              <span className="paper-card-date">{paper.publishDate}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
