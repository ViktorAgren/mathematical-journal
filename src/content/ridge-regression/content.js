import React from "react";
import { CollapsibleCodeWindow } from "../../components/CollapsibleCodeWindow";
import { LatexBlock } from "../../components/LatexBlock";
import { Theorem } from "../../components/Theorem";
import { Note } from "../../components/Note";
import { InlineMath } from "../../components/InlineMath";

export const RidgeRegressionContent = () => {
  return (
    <>
      <section className="paper-section">
        <h2 className="section-title">
          The $50 Million Portfolio Problem
        </h2>

        <p>
          Picture this: You're managing a $50 million equity portfolio, and your quantitative model just recommended going long Netflix and short Disney based on your "streaming wars" factor. Yesterday, the same model with 99% identical data suggested the exact opposite trade. Your investors are asking uncomfortable questions, and you're staring at a classic case of what financial engineers call "parameter instability". but what's really happening is far more fundamental.
        </p>

        <p>
          You're not dealing with a bug in your code. You're facing the central challenge of modern finance: <strong>multicollinearity</strong>. Netflix and Disney don't just happen to correlate. they're fighting the same battle, responding to the same consumer trends, competing for the same subscriber dollars. When you try to separate their individual effects using traditional regression, you're essentially asking: "If Netflix goes up 1% while Disney stays perfectly flat, what happens to the market?" It's a reasonable question with an impossible answer, because in the real world, Disney never stays flat when Netflix moves.
        </p>

        <p>
          Ridge regression doesn't solve this problem by being smarter about correlation. it solves it by being <em>humble</em>. Instead of pretending it can perfectly disentangle Netflix from Disney, it admits uncertainty and spreads the risk. It's the difference between a cocky trader who claims to know exactly which tech stock will outperform, and a wise investor who acknowledges that tech stocks rise and fall together, so maybe. just maybe. the smart money is on diversification.
        </p>

        <h2 className="section-title">
          Why Your Stock Predictions Keep Failing
        </h2>

        <p>
          Let's get concrete. You're building a model to predict S&P 500 returns using 20 different factors: sector momentum, earnings revisions, technical indicators, macro data. You have 250 trading days of data. about one year. In the machine learning world, this is a dream scenario: 250 samples, 20 features, clean data. Your model should work beautifully.
        </p>

        <p>
          Except it doesn't. Here's what actually happens:
        </p>

        <p>
          <strong>Monday's model:</strong> "Technology momentum is crucial (+0.8), healthcare earnings matter (+0.3), ignore bond yields (-0.1)."
        </p>

        <p>
          <strong>Tuesday's model (one new data point):</strong> "Healthcare earnings are everything (+0.9), technology momentum is noise (-0.2), bond yields are critical (+0.7)."
        </p>

        <p>
          What changed? Almost nothing in the market, but everything in your model. This isn't a failure of your analysis. it's the inevitable result of trying to solve an <em>ill-conditioned</em> problem. When your predictors are correlated (and in finance, they always are), ordinary least squares doesn't just give you the best answer. it gives you one of infinitely many answers that fit the data equally well.
        </p>

        <Note>
          <p>
            <strong>The Correlation Reality Check:</strong> In a typical equity factor model, correlations between predictors often exceed 0.7. Technology stocks correlate at 0.85+, sector momentum factors at 0.6+, and macroeconomic indicators show complex interdependencies. When predictors correlate above 0.9, ordinary least squares becomes practically unusable. your model will change dramatically with each new observation.
          </p>
        </Note>

        <h2 className="section-title">
          The Mathematical Heart of the Problem
        </h2>

        <p>
          The math here isn't abstract. it's the difference between profit and loss. Let's say you're trying to predict tomorrow's return <InlineMath tex="y" /> using today's factor exposures <InlineMath tex="x_1, x_2, \ldots, x_p" />. You believe the relationship follows:
        </p>

        <LatexBlock equation="y = \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p + \epsilon" />

        <p>
          Those <InlineMath tex="\beta" /> coefficients aren't just numbers. they're your portfolio weights, your risk allocations, your bet on which factors matter. Ordinary least squares finds the <InlineMath tex="\beta" /> values that minimize prediction error:
        </p>

        <LatexBlock equation="\hat{\beta}_{\text{OLS}} = \underset{\beta}{\arg\min} \sum_{i=1}^n (y_i - x_i^T \beta)^2" />

        <p>
          This has the elegant closed-form solution:
        </p>

        <LatexBlock equation="\hat{\beta}_{\text{OLS}} = (X^T X)^{-1} X^T y" />

        <p>
          The problem lurks in that innocent-looking inverse: <InlineMath tex="(X^T X)^{-1}" />. When your predictors are highly correlated. which they are in finance. this matrix becomes nearly <em>singular</em>. It's like trying to divide by a number very close to zero: mathematically possible, but practically disastrous.
        </p>

        <Theorem title="The Instability Theorem">
          <p>
            When predictors are highly correlated, the condition number of <InlineMath tex="X^T X" /> becomes large, meaning:
          </p>
          <LatexBlock equation="\text{condition number} = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1" />
          <p>
            Small changes in the data <InlineMath tex="X" /> or <InlineMath tex="y" /> lead to enormous changes in <InlineMath tex="\hat{\beta}" />. In financial terms: your model parameters swing wildly with each new market day.
          </p>
        </Theorem>

        <h2 className="section-title">
          Ridge Regression: The Humble Solution
        </h2>

        <p>
          Ridge regression solves the instability problem not by being more clever, but by being more conservative. It asks a slightly different question: "Instead of finding the coefficients that perfectly fit the training data, what if we find coefficients that fit reasonably well <em>and</em> aren't too extreme?"
        </p>

        <p>
          Mathematically, Ridge adds a penalty term that discourages large coefficients:
        </p>

        <LatexBlock equation="\hat{\beta}_{\text{Ridge}} = \underset{\beta}{\arg\min} \left\{ \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \lambda \sum_{j=1}^p \beta_j^2 \right\}" />

        <p>
          That <InlineMath tex="\lambda" /> parameter is your dial between "aggressive fitting" (<InlineMath tex="\lambda = 0" />, which gives you OLS) and "conservative regularization" (large <InlineMath tex="\lambda" />). The beauty is in the closed-form solution:
        </p>

        <LatexBlock equation="\hat{\beta}_{\text{Ridge}} = (X^T X + \lambda I)^{-1} X^T y" />

        <p>
          By adding <InlineMath tex="\lambda I" /> to the matrix, Ridge regression ensures that <InlineMath tex="X^T X + \lambda I" /> is always invertible, even when <InlineMath tex="X^T X" /> isn't. It's like adding a small amount of sand to a slippery surface. it provides the traction needed for stability.
        </p>

        <h2 className="section-title">
          The Bias-Variance Tradeoff in Your Portfolio
        </h2>

        <p>
          Here's where Ridge regression gets philosophically interesting for investors. Traditional statistics teaches us that unbiased estimators are good. we want our predictions to be "correct on average." But in finance, being unbiased isn't worth much if you're also wildly inconsistent.
        </p>

        <p>
          Ridge regression deliberately introduces <em>bias</em> to reduce <em>variance</em>. In portfolio terms: it's willing to be slightly wrong on average if it means being much more stable day-to-day. This isn't a bug. it's the core insight.
        </p>

        <Theorem title="Bias-Variance Decomposition">
          <p>
            The expected prediction error of any estimator decomposes as:
          </p>
          <LatexBlock equation="\mathbb{E}[\text{MSE}] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}" />
          <p>
            Ridge regression increases bias slightly but can dramatically reduce variance, often resulting in lower total error.
          </p>
        </Theorem>

        <p>
          In practical terms, Ridge regression gives you a portfolio model that:
        </p>
        <ul style={{ marginLeft: '2rem', marginBottom: '1.5rem' }}>
          <li><strong>Changes gradually</strong> as new data arrives (low variance)</li>
          <li><strong>Spreads risk</strong> across correlated factors rather than betting everything on one</li>
          <li><strong>Admits uncertainty</strong> by shrinking extreme coefficient estimates toward zero</li>
          <li><strong>Performs consistently</strong> out-of-sample, even if not perfectly in-sample</li>
        </ul>

        <h2 className="section-title">
          Real-World Implementation: Building a Sector Rotation Model
        </h2>

        <p>
          Let's build something you could actually use tomorrow. Here's a complete Ridge regression implementation for predicting sector returns based on macroeconomic factors and cross-sector momentum:
        </p>

        <CollapsibleCodeWindow
          language="python"
          title="Ridge Regression for Sector Rotation Strategy"
          code={`
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

def build_sector_rotation_model():
    """
    Build a Ridge regression model for sector rotation based on 
    macroeconomic factors and momentum indicators.
    
    Returns sector allocation recommendations with confidence intervals.
    """
    
    # Download sector ETF data
    sectors = {
        'XLK': 'Technology',
        'XLF': 'Financials', 
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrials',
        'XLY': 'Consumer Discretionary',
        'XLP': 'Consumer Staples',
        'XLRE': 'Real Estate',
        'XLU': 'Utilities'
    }
    
    # Get 5 years of data
    sector_data = yf.download(list(sectors.keys()), 
                             start="2019-01-01", 
                             end="2024-01-01")['Adj Close']
    
    # Calculate returns
    sector_returns = sector_data.pct_change().dropna()
    
    # Create feature matrix
    features = pd.DataFrame(index=sector_returns.index)
    
    # 1. Momentum features (past returns)
    for days in [5, 20, 60]:
        momentum = sector_data.pct_change(days)
        for sector in sectors.keys():
            features[f'{sector}_mom_{days}d'] = momentum[sector]
    
    # 2. Volatility features  
    for days in [20, 60]:
        vol = sector_returns.rolling(days).std()
        for sector in sectors.keys():
            features[f'{sector}_vol_{days}d'] = vol[sector]
    
    # 3. Relative strength features
    spy_data = yf.download('SPY', start="2019-01-01", end="2024-01-01")['Adj Close']
    spy_returns = spy_data.pct_change()
    
    for sector in sectors.keys():
        # Relative performance vs SPY
        features[f'{sector}_rel_perf'] = (
            sector_returns[sector] - spy_returns
        ).rolling(60).mean()
    
    # 4. Cross-sector correlation (multicollinearity indicator)
    corr_window = 60
    for i, sector1 in enumerate(sectors.keys()):
        for sector2 in list(sectors.keys())[i+1:]:
            rolling_corr = sector_returns[sector1].rolling(corr_window).corr(
                sector_returns[sector2]
            )
            features[f'corr_{sector1}_{sector2}'] = rolling_corr
    
    # Clean features
    features = features.dropna()
    
    # Align returns with features
    aligned_returns = sector_returns.loc[features.index]
    
    print(f"Dataset shape: {features.shape}")
    print(f"Feature correlation stats:")
    corr_matrix = features.corr()
    print(f"  - Max correlation: {corr_matrix.max().max():.3f}")
    print(f"  - Min correlation: {corr_matrix.min().min():.3f}")
    print(f"  - Mean abs correlation: {np.abs(corr_matrix).mean().mean():.3f}")
    
    # Time series cross-validation setup
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}
    
    # Analyze each sector
    for sector_name, sector_desc in sectors.items():
        print(f"\nAnalyzing {sector_desc} ({sector_name})...")
        
        # Target: next period return
        y = aligned_returns[sector_name].shift(-1).dropna()
        X = features.loc[y.index]
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Ridge regression with cross-validation
        alphas = np.logspace(-4, 2, 50)
        ridge_cv = RidgeCV(
            alphas=alphas,
            cv=tscv,
            scoring='neg_mean_squared_error',
            store_cv_values=True
        )
        
        ridge_cv.fit(X_scaled, y)
        optimal_alpha = ridge_cv.alpha_
        
        print(f"  Optimal λ: {optimal_alpha:.6f}")
        
        # Compare Ridge vs OLS performance
        ridge_scores = []
        ols_scores = []
        feature_stability = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Ridge model
            ridge = Ridge(alpha=optimal_alpha)
            ridge.fit(X_train, y_train)
            ridge_pred = ridge.predict(X_test)
            ridge_scores.append(r2_score(y_test, ridge_pred))
            
            # Track coefficient stability
            feature_stability.append(np.std(ridge.coef_))
            
            # OLS comparison (Ridge with λ=0.0001 ≈ OLS)
            ols = Ridge(alpha=0.0001)
            ols.fit(X_train, y_train)
            ols_pred = ols.predict(X_test)
            ols_scores.append(r2_score(y_test, ols_pred))
        
        # Performance summary
        ridge_mean = np.mean(ridge_scores)
        ridge_std = np.std(ridge_scores)
        ols_mean = np.mean(ols_scores)
        ols_std = np.std(ols_scores)
        
        print(f"  Ridge R²: {ridge_mean:.4f} ± {ridge_std:.4f}")
        print(f"  OLS R²:   {ols_mean:.4f} ± {ols_std:.4f}")
        print(f"  Stability improvement: {(ols_std/ridge_std):.2f}x")
        
        # Final model on full dataset
        final_model = Ridge(alpha=optimal_alpha)
        final_model.fit(X_scaled, y)
        
        # Feature importance (top 10)
        feature_importance = pd.Series(
            final_model.coef_, 
            index=X.columns
        ).abs().sort_values(ascending=False)
        
        print(f"  Top 5 features:")
        for feat, importance in feature_importance.head().items():
            print(f"    {feat}: {importance:.4f}")
        
        results[sector_name] = {
            'model': final_model,
            'scaler': scaler,
            'features': X.columns,
            'performance': {
                'ridge_r2_mean': ridge_mean,
                'ridge_r2_std': ridge_std,
                'ols_r2_mean': ols_mean,
                'ols_r2_std': ols_std,
                'stability_ratio': ols_std/ridge_std
            },
            'lambda': optimal_alpha
        }
    
    return results, features, aligned_returns

def generate_sector_predictions(results, current_features):
    """
    Generate next-period sector return predictions with confidence intervals.
    """
    predictions = {}
    
    for sector, model_info in results.items():
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Scale current features
        current_scaled = scaler.transform([current_features])
        
        # Predict
        pred = model.predict(current_scaled)[0]
        
        # Rough confidence interval (assuming normal errors)
        # In practice, you'd use bootstrap or cross-validation residuals
        std_error = np.sqrt(model_info['performance']['ridge_r2_std'])
        
        predictions[sector] = {
            'prediction': pred,
            'lower_ci': pred - 1.96 * std_error,
            'upper_ci': pred + 1.96 * std_error,
            'lambda': model_info['lambda']
        }
    
    return predictions

# Demonstration
if __name__ == "__main__":
    print("Building Sector Rotation Model with Ridge Regression...")
    print("=" * 60)
    
    # Build models
    results, features, returns = build_sector_rotation_model()
    
    print("\n" + "=" * 60)
    print("Model Performance Summary:")
    print("=" * 60)
    
    stability_improvements = []
    for sector, info in results.items():
        perf = info['performance']
        stability_improvements.append(perf['stability_ratio'])
        print(f"{sector}: Stability improved by {perf['stability_ratio']:.2f}x")
    
    avg_stability = np.mean(stability_improvements)
    print(f"\nAverage stability improvement: {avg_stability:.2f}x")
    print(f"This means Ridge coefficients are {avg_stability:.1f}x more stable than OLS!")
    
    # Generate predictions for last available data point
    print("\n" + "=" * 60)
    print("Sample Predictions (last data point):")
    print("=" * 60)
    
    last_features = features.iloc[-1]
    predictions = generate_sector_predictions(results, last_features)
    
    # Sort by prediction
    sorted_preds = sorted(predictions.items(), 
                         key=lambda x: x[1]['prediction'], 
                         reverse=True)
    
    for sector, pred_info in sorted_preds:
        pred = pred_info['prediction']
        ci_lower = pred_info['lower_ci']
        ci_upper = pred_info['upper_ci']
        print(f"{sector}: {pred:+.3f} [{ci_lower:+.3f}, {ci_upper:+.3f}]")
          `}
        />

        <h2 className="section-title">
          Advanced Applications: Beyond Basic Ridge
        </h2>

        <p>
          The sector rotation model above is just the beginning. Here are three advanced applications where Ridge regression shines in finance:
        </p>

        <h3 className="subsection-title">1. High-Frequency Mean Reversion</h3>
        <p>
          In intraday trading, price movements are heavily autocorrelated across short intervals. Ridge regression helps build stable mean-reversion models that don't overfit to noise in tick-by-tick data.
        </p>

        <h3 className="subsection-title">2. Multi-Asset Portfolio Optimization</h3>
        <p>
          When optimizing portfolios with hundreds of assets, the sample covariance matrix is notoriously unstable. Ridge-regularized covariance estimation (using <InlineMath tex="\hat{\Sigma} + \lambda I" />) produces more robust portfolio weights.
        </p>

        <h3 className="subsection-title">3. Factor Model Construction</h3>
        <p>
          Building custom risk factors from fundamental data involves regression on highly correlated predictors (think: all the different profitability ratios). Ridge regression ensures your factor loadings are stable and economically interpretable.
        </p>

        <Note>
          <p>
            <strong>Production Tip:</strong> In live trading systems, monitor the condition number of your feature matrix daily. When it exceeds 1000, increase your Ridge penalty. When it's below 10, you might be over-regularizing. The sweet spot for most financial applications is a condition number between 20-100.
          </p>
        </Note>

        <h2 className="section-title">
          The Deeper Truth About Financial Modeling
        </h2>

        <p>
          Ridge regression teaches us something profound about financial modeling: the goal isn't to find the "true" model. it's to find a <em>useful</em> model. In a world where Netflix and Disney move together, pretending you can perfectly separate their individual effects is hubris. Admitting that you can't, and building models that account for this uncertainty, is wisdom.
        </p>

        <p>
          This philosophical shift changes everything. Instead of asking "What's the exact beta of healthcare stocks to interest rates?" Ridge regression asks "Given that healthcare, utilities, and REITs all respond to rates in overlapping ways, what's a reasonable estimate that won't blow up when the data changes slightly?" It's the difference between precision and accuracy, between fitting and generalizing, between looking smart on paper and making money in reality.
        </p>

        <p>
          When your Ridge-regularized model tells you that technology momentum has a coefficient of 0.3 instead of 0.8, it's not being imprecise. it's being honest about the uncertainty inherent in financial markets. That honesty is what keeps your portfolio stable when correlations shift, volatility spikes, and the market reminds everyone that past performance doesn't guarantee future results.
        </p>

        <Theorem title="The Regularization Principle in Finance">
          <p>
            In financial modeling, the optimal amount of regularization <InlineMath tex="\lambda^*" /> balances three competing forces:
          </p>
          <LatexBlock equation="\lambda^* = \arg\min_\lambda \{\text{Fitting Error} + \text{Complexity Penalty} + \text{Stability Cost}\}" />
          <p>
            Too little regularization leads to overfitting and parameter instability. Too much leads to underfitting and missed opportunities. The optimal <InlineMath tex="\lambda^*" /> depends on your data's correlation structure, noise level, and the economic cost of model instability.
          </p>
        </Theorem>

        <p>
          The next time someone shows you a financial model with suspiciously precise coefficients that change dramatically each month, you'll know what's missing: a healthy dose of regularization, a bit of humility about what can actually be predicted in markets, and the wisdom to choose stability over the illusion of precision.
        </p>

        <p>
          Ridge regression doesn't just solve a statistical problem. it embodies a philosophy of robust decision-making under uncertainty. In finance, that philosophy isn't just mathematically elegant; it's profitable.
        </p>
      </section>
    </>
  );
};
