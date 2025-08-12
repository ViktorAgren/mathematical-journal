import React from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

export const CodeWindow = ({ code, language = "python", title }) => {
  return (
    <div className="code-block">
      {title && (
        <div className="code-header">
          <h4 className="code-title">{title}</h4>
          <span className="code-language">{language}</span>
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
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
};
