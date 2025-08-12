// Content configuration for mathematical papers
import { RidgeRegressionContent } from "./ridge-regression/content";
import { SoccerPositionModelsContent } from "./soccer-position-models/content";
import { PitchControlContent } from "./pitch-control/content";

export const papers = {
  "ridge-regression": {
    id: "ridge-regression",
    title: "Ridge Regression in Finance",
    subtitle: "Taming market volatility through regularization",
    publishDate: "2024-12-10",
    tags: ["Quantitative Finance", "Statistics", "Portfolio Theory"],
    abstract:
      "Discover how ridge regression transforms chaotic financial data into stable predictions. From tackling multicollinearity in factor models to optimizing portfolios with hundreds of stocks, learn why Wall Street relies on this regularization technique to navigate market uncertainty and build robust trading strategies.",
    content: RidgeRegressionContent,
  },
  "soccer-position-models": {
    id: "soccer-position-models",
    title: "Soccer Position Value Models",
    subtitle: "Mathematical frameworks for player valuation",
    publishDate: "2025-04-11",
    tags: ["Sports Analytics", "Statistics", "Machine Learning"],
    abstract:
      "A comprehensive mathematical treatment of player valuation in soccer. Covers expected goals models, position-specific metrics, market value prediction, and advanced analytics including expected threat and zone control. Demonstrates practical implementation for transfer market analysis and squad optimization.",
    content: SoccerPositionModelsContent,
  },
  "pitch-control": {
    id: "pitch-control",
    title: "Pitch Control Theory",
    subtitle: "Geometric deep learning for tactical analysis",
    publishDate: "2025-06-01",
    tags: ["Sports Analytics", "Deep Learning", "Geometric AI"],
    abstract:
      "Explore how Liverpool FC and DeepMind revolutionized football tactics through geometric deep learning. From spatial optimization to corner kick strategies, discover the mathematical framework that models territorial control and reveals the invisible tactical battles that determine match outcomes.",
    content: PitchControlContent,
  },
};

export const getPaper = (paperId) => papers[paperId];
export const getAllPapers = () => Object.values(papers);
