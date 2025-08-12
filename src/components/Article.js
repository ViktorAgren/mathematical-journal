import React from 'react';

export const Article = ({ paper }) => {
  if (!paper) {
    return (
      <div className="article-error">
        <h1>Paper not found</h1>
        <p>The requested paper could not be loaded.</p>
      </div>
    );
  }

  const ContentComponent = paper.content;

  return (
    <article className="journal-article">
      <header className="article-header">
        <div className="article-meta">
          <div className="article-tags">
            {paper.tags.map((tag) => (
              <span key={tag} className="article-tag">
                {tag}
              </span>
            ))}
          </div>
          <div className="article-info">
            <span className="article-reading-time">{paper.readingTime}</span>
            <span className="article-difficulty">{paper.difficulty}</span>
            <span className="article-date">{paper.publishDate}</span>
          </div>
        </div>
        
        <h1 className="article-title">{paper.title}</h1>
        <p className="article-subtitle">{paper.subtitle}</p>
        <p className="article-author">by {paper.author}</p>
        
        <div className="article-abstract">
          <p>{paper.abstract}</p>
        </div>
      </header>

      <div className="article-content">
        <ContentComponent />
      </div>
    </article>
  );
};