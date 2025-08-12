import React, { useState } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export const CollapsibleCodeWindow = ({ 
  code, 
  language = "python", 
  title, 
  previewLines = 20,
  showExpandButton = true 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  const codeLines = code.split('\n');
  const shouldCollapse = showExpandButton && codeLines.length > previewLines + 5;
  const displayCode = shouldCollapse && !isExpanded 
    ? codeLines.slice(0, previewLines).join('\n') + '\n\n# ... (code continues below)'
    : code;

  const toggleExpanded = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="code-block collapsible-code">
      {title && (
        <div className="code-header">
          <h4 className="code-title">{title}</h4>
          <div className="code-header-right">
            <span className="code-language">{language}</span>
            {shouldCollapse && (
              <button 
                onClick={toggleExpanded}
                className="code-expand-btn"
                aria-label={isExpanded ? "Collapse code" : "Expand code"}
              >
                {isExpanded ? (
                  <>
                    <svg className="code-expand-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                    </svg>
                    Show Less
                  </>
                ) : (
                  <>
                    <svg className="code-expand-icon" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                    View Complete Implementation ({codeLines.length} lines)
                  </>
                )}
              </button>
            )}
          </div>
        </div>
      )}
      <div className="code-content">
        <SyntaxHighlighter
          language={language}
          style={oneDark}
          customStyle={{
            margin: 0,
            padding: '1.5rem',
            background: 'var(--bg-code)',
            fontSize: '0.875rem',
            lineHeight: '1.5',
          }}
          showLineNumbers={true}
          lineNumberStyle={{
            color: '#64748b',
            fontSize: '0.75rem',
            minWidth: '3em',
            paddingRight: '1em',
          }}
        >
          {displayCode}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};