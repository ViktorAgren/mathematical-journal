# Mathematical Journal

A web-based platform for publishing mathematical papers with interactive code examples and LaTeX rendering. Built with React 18 and designed for academic-quality mathematical content presentation.

## Current Papers

- **Ridge Regression in Finance**: Regularization techniques for portfolio optimization and market volatility analysis
- **Soccer Position Value Models**: Mathematical frameworks for player valuation using expected possession value and spatial analytics  
- **Pitch Control Theory**: Geometric deep learning approaches to tactical analysis based on Liverpool FC/DeepMind research

## Installation

```bash
npm install
```

## Development

```bash
npm start
```

Opens development server at http://localhost:3000

## Production Build

```bash
npm run build
```

## GitHub Pages Deployment

```bash
npm run deploy
```

## Project Structure

```
src/
├── components/          # Reusable UI components (LatexBlock, CodeWindow, etc.)
├── content/            # Paper content and configuration
│   ├── papers.js       # Paper metadata and routing
│   ├── ridge-regression/
│   ├── soccer-position-models/
│   └── pitch-control/
└── styles/             # CSS styling system
```

## Adding New Papers

1. Create content directory in `src/content/[paper-name]/`
2. Add `content.js` with paper component
3. Update `src/content/papers.js` with metadata

## Technical Stack

- React 18 with functional components
- MathJax 3 for LaTeX mathematical notation
- React Syntax Highlighter for code blocks
- CSS Grid and Flexbox for responsive layout
- GitHub Pages for deployment
