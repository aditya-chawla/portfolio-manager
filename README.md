# Hierarchical Reinforcement Learning for Portfolio Management 📈

**Advanced Multi-Tier RL Architecture for Automated Trading**

[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-green.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Stable--Baselines3-orange.svg)](https://stable-baselines3.readthedocs.io/)

> A three-tier hierarchical reinforcement learning system that combines traditional market data with Reddit sentiment analysis to achieve 74.79% annualized returns with superior risk-adjusted performance.

---

## 🎯 Overview

This project implements a novel hierarchical RL architecture for portfolio management that significantly outperforms traditional approaches. By training multiple RL algorithms on dual information streams (market data + social sentiment), the system achieved:

- **74.79% Annualized Return** vs. 28.13% for equal-weighted baseline
- **2.89 Sharpe Ratio** vs. 1.58 for diversified portfolio
- **-9.86% Maximum Drawdown** vs. -15.62% for baseline
- **Superior Risk-Adjusted Performance** across all metrics

### Key Innovation

Unlike single-agent approaches, our three-tier architecture:
1. **Base Tier** - 8 agents (PPO, SAC, DDPG, TD3) trained on separate data streams
2. **Meta Tier** - 2 aggregation agents combine base strategies per modality
3. **Super Tier** - 1 integration agent learns when to favor each information source

This hierarchical design allows the system to adapt dynamically to changing market conditions by leveraging the strengths of different algorithms and information sources.

---

## 📊 Performance Highlights

### Returns Comparison

| Strategy | Total ROI | Annual ROI | Sharpe Ratio | Max Drawdown |
|----------|-----------|------------|--------------|--------------|
| **Hierarchical RL (Super-Agent)** | **55.07%** | **74.79%** | **2.89** | **-9.86%** |
| Equal-Weighted Baseline | 25.26% | 28.13% | 1.58 | -15.62% |
| Best Base Agent (DDPG) | 21.48% | 28.10% | 1.55 | -12.14% |
| S&P 500 Only | 16.71% | 18.53% | 0.97 | -18.90% |

### Risk-Adjusted Performance

![Sharpe Ratio Comparison](docs/images/sharpe_comparison.png)

The super-agent achieves **2.66x better returns** than the best individual algorithm while maintaining **lower volatility** and **reduced downside risk**.

---

## 🏗️ Architecture

### Three-Tier Hierarchical Design

```
┌─────────────────────────────────────────────────────────────┐
│                        SUPER-AGENT                          │
│              (PPO - Final Portfolio Integration)            │
│    Combines Meta-Data + Meta-NLP + Market Regime           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                ┌─────────────┴─────────────┐
                │                           │
┌───────────────┴─────────────┐  ┌─────────┴──────────────────┐
│      META-AGENT (DATA)      │  │     META-AGENT (NLP)       │
│  PPO - Aggregates 4 agents  │  │  PPO - Aggregates 4 agents │
│   84-dim obs (28+56 feat)   │  │   63-dim obs (28+35 feat)  │
└───────────────┬─────────────┘  └─────────┬──────────────────┘
                │                          │
    ┌───────────┼───────────┐  ┌──────────┼──────────┐
    │           │           │  │          │          │
┌───▼───┐ ┌───▼───┐ ┌────▼──┐ ┌▼────┐ ┌──▼───┐ ┌───▼──┐
│  PPO  │ │  SAC  │ │ DDPG  │ │ TD3 │ │ PPO  │ │ SAC  │ ...
│ Data  │ │ Data  │ │ Data  │ │Data │ │ NLP  │ │ NLP  │
└───────┘ └───────┘ └───────┘ └─────┘ └──────┘ └──────┘
   56-dim      56-dim     56-dim    56-dim   35-dim   35-dim
  Market       Market     Market    Market   Sent.    Sent.
  Features     Features   Features  Features  Feat.    Feat.
```

### Information Streams

**Data-Driven Features (56-dim):**
- Sharpe Ratio, Calmar Ratio, Sortino Ratio
- Annualized Volatility, Maximum Drawdown
- Pairwise Asset Correlations (21 unique pairs)

**Sentiment-Driven Features (35-dim):**
- FinBERT Sentiment Score (positive/negative balance)
- Sentiment Volatility (opinion divergence)
- Article Frequency (discussion volume)
- Model Confidence Score
- Volatility Signal (uncertainty flag)

---

## 🛠️ Tech Stack

### Core ML/RL
- **Stable-Baselines3** - RL algorithm implementations (PPO, SAC, DDPG, TD3)
- **PyTorch** - Neural network backend
- **Gymnasium** - Custom RL environments

### Data & NLP
- **Yahoo Finance API** - Historical market data (prices, volumes)
- **PRAW (Reddit API)** - Social media data collection
- **FinBERT** - Financial sentiment analysis (fine-tuned BERT)
- **Pandas & NumPy** - Data processing

### Portfolio Assets (7 Assets)
- **US Equities:** S&P 500, NASDAQ, Dow Jones, Russell 2000
- **Commodities:** Gold Futures, Silver Futures, Crude Oil (WTI)

---

## 📁 Project Structure

```
portfolio-manager/
├── notebooks/
│   ├── RL_with_NLP_Meta_data.ipynb             # 🎯 MAIN: Complete implementation
│   ├── Complete_HRL_with_NLP_Meta_data.ipynb   # Full training pipeline
│   ├── Meta_data_with_base_agents.ipynb        # Data-driven agents only
│   └── NLP_for_RL_Meta_data.ipynb              # Sentiment agents only
├── data/
│   ├── market/
│   │   ├── prices_2022_2025.csv                # Yahoo Finance data
│   │   └── returns_2022_2025.csv               # Calculated returns
│   └── sentiment/
│       ├── reddit_posts_stocks.json            # r/stocks posts
│       ├── reddit_posts_investing.json         # r/investing posts
│       ├── reddit_posts_wsb.json               # r/wallstreetbets posts
│       └── sentiment_features.csv              # Processed FinBERT features
├── models/
│   ├── base_agents/
│   │   ├── ppo_data.zip                        # PPO on market data
│   │   ├── sac_data.zip                        # SAC on market data
│   │   ├── ddpg_data.zip                       # DDPG on market data
│   │   ├── td3_data.zip                        # TD3 on market data
│   │   ├── ppo_nlp.zip                         # PPO on sentiment
│   │   ├── sac_nlp.zip                         # SAC on sentiment
│   │   ├── ddpg_nlp.zip                        # DDPG on sentiment
│   │   └── td3_nlp.zip                         # TD3 on sentiment
│   ├── meta_agents/
│   │   ├── meta_data.zip                       # Data stream aggregator
│   │   └── meta_nlp.zip                        # Sentiment stream aggregator
│   └── super_agent/
│       └── super_agent_best.zip                # Final integration agent
├── results/
│   ├── performance_metrics.csv                 # All agents' metrics
│   ├── allocation_history.csv                  # Daily portfolio weights
│   ├── cumulative_returns.png                  # Performance chart
│   ├── sharpe_comparison.png                   # Risk-adjusted returns
│   └── allocation_heatmap.png                  # Asset allocation over time
├── docs/
│   ├── RL_Paper_WIP.pdf                        # Research paper
│   └── images/
│       ├── architecture_diagram.png
│       ├── sharpe_comparison.png
│       └── cumulative_returns.png
├── requirements.txt                             # Python dependencies
├── .env.example                                 # Environment template
├── LICENSE                                      # Apache-2.0
└── README.md                                    # This file
```

### Main Notebook Structure

The `RL_with_NLP_Meta_data.ipynb` notebook contains:

1. **Setup & Imports** (Cells 1-4)
   - Library imports (stable-baselines3, gymnasium, yfinance, transformers)
   - Configuration and constants
   
2. **Data Collection** (Cells 5-7)
   - Yahoo Finance API calls for market data
   - Reddit scraping with asyncpraw
   - Date range: 2022-2025 (3 years train, 11 months test)

3. **Feature Engineering** (Cells 8-11)
   - Market features: Sharpe, Sortino, Calmar, volatility, correlations
   - Sentiment features: FinBERT processing, sentiment scores
   - Rolling window calculations (30-day windows)

4. **Environment Classes** (Cells 12-16)
   - `BasePortfolioEnv`: Base agents (56-dim or 35-dim obs)
   - `MetaAgentEnv`: Meta aggregation (84-dim or 63-dim obs)
   - `SuperAgentEnv`: Final integration (15-dim obs)

5. **Base Agent Training** (Cells 17-20)
   - 8 agents total: 4 data-driven + 4 sentiment-driven
   - Algorithms: PPO, SAC, DDPG, TD3
   - 30,000 timesteps each (~15 min per agent on Colab)

6. **Meta-Agent Training** (Cells 21-23)
   - Meta-Data: Aggregates 4 data-driven agents
   - Meta-NLP: Aggregates 4 sentiment-driven agents
   - 15,000 timesteps each

7. **Super-Agent Training** (Cells 24-25)
   - Combines both meta-agents
   - 10,000 timesteps
   - Learns stream weighting strategy

8. **Evaluation & Results** (Cells 26-29)
   - Test period evaluation (Jan-Nov 2025)
   - Performance metrics calculation
   - Visualization: returns, allocations, comparisons
   - Baseline comparisons

---

## 🚀 Getting Started

### Prerequisites

```bash
- Python >= 3.9
- Jupyter Notebook
- Reddit API credentials (for sentiment collection)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shyam-kannan/portfolio-manager.git
cd portfolio-manager
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Required packages:**
```txt
# Core ML/RL
stable-baselines3[extra]==2.0.0
gymnasium==0.29.0
torch>=2.0.0

# Data & Finance
yfinance==0.2.28
pandas>=1.5.0
numpy>=1.24.0

# NLP & Sentiment
transformers==4.30.0
praw==7.7.0
asyncpraw==7.7.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utils
python-dotenv==1.0.0
```

3. **Set up Reddit API credentials**

Create a Reddit app at https://www.reddit.com/prefs/apps

```bash
# Create .env file
cat > .env << EOF
REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_secret_here
REDDIT_USER_AGENT=YourApp/1.0
EOF
```

4. **Download FinBERT model**
```python
from transformers import BertTokenizer, BertForSequenceClassification

# This will download automatically on first use
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
```

---

## 💬 Reddit Sentiment Collection

### Collecting Social Media Data

```python
import asyncpraw
from transformers import BertTokenizer, BertForSequenceClassification
import torch

async def collect_reddit_sentiment():
    """Collect posts from investment subreddits"""
    
    # Initialize Reddit API
    reddit = asyncpraw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        user_agent=os.getenv('REDDIT_USER_AGENT')
    )
    
    # Target subreddits
    subreddits = ['stocks', 'investing', 'wallstreetbets', 'StockMarket']
    
    posts = []
    for sub_name in subreddits:
        subreddit = await reddit.subreddit(sub_name)
        async for post in subreddit.hot(limit=500):
            posts.append({
                'title': post.title,
                'text': post.selftext,
                'score': post.score,
                'created': post.created_utc,
                'subreddit': sub_name
            })
    
    return posts

# Asset-specific keywords for filtering
ASSET_KEYWORDS = {
    'S&P 500': ['S&P 500', 'SPY', 'SP500', 'S&P', 'VOO'],
    'NASDAQ': ['NASDAQ', 'QQQ', 'tech stocks'],
    'DOW': ['Dow Jones', 'DIA', 'DJIA'],
    'Russell': ['Russell 2000', 'IWM', 'small cap'],
    'Gold': ['gold', 'GLD', 'gold prices'],
    'Silver': ['silver', 'SLV', 'silver prices'],
    'Oil': ['oil', 'crude oil', 'USO', 'WTI']
}

def filter_posts_by_asset(posts: List[dict], asset: str) -> List[dict]:
    """Filter posts relevant to specific asset"""
    keywords = ASSET_KEYWORDS.get(asset, [])
    relevant_posts = []
    
    for post in posts:
        text = (post['title'] + ' ' + post['text']).lower()
        if any(keyword.lower() in text for keyword in keywords):
            relevant_posts.append(post)
    
    return relevant_posts

def analyze_sentiment_finbert(texts: List[str]):
    """Analyze sentiment using FinBERT"""
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    
    sentiments = []
    for text in texts:
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
        
        # probs: [negative, neutral, positive]
        sentiments.append({
            'negative': probs[0],
            'neutral': probs[1],
            'positive': probs[2],
            'score': probs[2] - probs[0]  # Net sentiment
        })
    
    return sentiments
```

---

## 📖 Usage

### Quick Start - Run the Complete System

Open the main notebook to train and evaluate the full hierarchical system:

```bash
jupyter notebook RL_with_NLP_Meta_data.ipynb
```

### Step-by-Step Training

#### 1. Set Up Data Collection

```python
import yfinance as yf
import pandas as pd
from datetime import datetime

# Define portfolio assets
tickers = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC', 
    'DOW': '^DJI',
    'Russell': '^RUT',
    'Gold': 'GC=F',
    'Silver': 'SI=F',
    'Oil': 'CL=F'
}

# Fetch historical data
start_date = '2022-01-01'
end_date = '2025-11-30'

prices = yf.download(list(tickers.values()), start=start_date, end=end_date)['Close']
```

#### 2. Feature Engineering

```python
def compute_data_features(prices: pd.DataFrame, window: int = 30):
    """Compute 56-dimensional market features"""
    returns = prices.pct_change().dropna()
    features_list = []
    
    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        
        # Sharpe Ratio (7 assets)
        sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
        
        # Sortino Ratio (7 assets)
        downside_returns = window_returns[window_returns < 0]
        sortino = window_returns.mean() / downside_returns.std()
        
        # Calmar Ratio (7 assets)
        cumulative = (1 + window_returns).cumprod()
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        calmar = window_returns.mean() * 252 / abs(max_dd)
        
        # Volatility (7 assets)
        volatility = window_returns.std() * np.sqrt(252)
        
        # Maximum Drawdown (7 assets)
        max_drawdown = max_dd
        
        # Correlation matrix (21 unique pairs)
        corr_matrix = window_returns.corr()
        correlations = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        
        # Combine: 5 metrics × 7 assets + 21 correlations = 56 features
        features = np.concatenate([
            sharpe.values, sortino.values, calmar.values,
            volatility.values, max_drawdown.values, correlations
        ])
        features_list.append(features)
    
    return np.array(features_list)

def compute_nlp_features(reddit_posts: List[dict], assets: List[str]):
    """Compute 35-dimensional sentiment features using FinBERT"""
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch
    
    # Load FinBERT
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone')
    
    features_list = []
    
    for asset in assets:
        # Filter posts by keywords
        relevant_posts = filter_posts_by_asset(reddit_posts, asset)
        
        # Sentiment scores
        sentiments = []
        for post in relevant_posts:
            inputs = tokenizer(post['text'], return_tensors='pt', truncation=True)
            outputs = model(**inputs)
            sentiment = torch.softmax(outputs.logits, dim=1).detach().numpy()[0]
            sentiments.append(sentiment)
        
        # Aggregate features (5 per asset)
        sentiment_score = np.mean([s[1] - s[0] for s in sentiments])  # positive - negative
        sentiment_volatility = np.std([s[1] - s[0] for s in sentiments])
        article_frequency = len(relevant_posts)
        confidence = np.mean([np.max(s) for s in sentiments])
        volatility_signal = int(sentiment_volatility > threshold)
        
        features_list.extend([
            sentiment_score, sentiment_volatility, article_frequency,
            confidence, volatility_signal
        ])
    
    return np.array(features_list)  # 5 features × 7 assets = 35-dim
```

#### 3. Create Custom Gymnasium Environment

```python
import gymnasium as gym
from gymnasium import spaces

class BasePortfolioEnv(gym.Env):
    """Custom environment for base RL agents"""
    
    def __init__(self, prices: pd.DataFrame, features: np.ndarray, 
                 feature_type: str = 'data'):
        super().__init__()
        
        self.prices = prices.values
        self.returns = prices.pct_change().fillna(0).values
        self.features = features
        self.n_assets = prices.shape[1]
        self.current_step = 0
        
        # Action space: portfolio weights [0,1]^7 that sum to 1
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )
        
        # Observation space: 56-dim for data, 35-dim for NLP
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(features.shape[1],), dtype=np.float32
        )
        
        self.portfolio_value = 100000  # Start with $100k
    
    def step(self, action):
        # Normalize weights to sum to 1
        weights = action / (action.sum() + 1e-8)
        
        # Calculate portfolio return
        asset_returns = self.returns[self.current_step]
        portfolio_return = np.dot(weights, asset_returns)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Calculate reward: return - 0.5*volatility - 2.0*drawdown
        reward = self._compute_reward(portfolio_return, weights)
        
        self.current_step += 1
        done = self.current_step >= len(self.features)
        
        obs = self.features[self.current_step] if not done else self.features[-1]
        
        return obs, reward, done, False, {}
    
    def _compute_reward(self, portfolio_return, weights):
        # Get recent returns for volatility calculation
        recent_returns = self.returns[max(0, self.current_step-30):self.current_step]
        volatility = np.std(recent_returns @ weights) if len(recent_returns) > 0 else 0
        
        # Calculate drawdown
        peak_value = np.max(self.portfolio_values[:self.current_step+1])
        drawdown = (peak_value - self.portfolio_value) / peak_value
        
        # Reward function
        reward = portfolio_return - 0.5 * volatility - 2.0 * drawdown
        return reward
```

#### 4. Train Base Agents

```python
from stable_baselines3 import PPO, SAC, DDPG, TD3

# Prepare environments
data_env = DummyVecEnv([lambda: BasePortfolioEnv(prices, data_features, 'data')])
nlp_env = DummyVecEnv([lambda: BasePortfolioEnv(prices, nlp_features, 'nlp')])

# Train 4 algorithms on data stream
base_agents_data = {}
for algo_name, algo_class in [('PPO', PPO), ('SAC', SAC), ('DDPG', DDPG), ('TD3', TD3)]:
    print(f"Training {algo_name} (Data)...")
    agent = algo_class('MlpPolicy', data_env, verbose=1)
    agent.learn(total_timesteps=30000)
    base_agents_data[algo_name] = agent

# Train 4 algorithms on NLP stream
base_agents_nlp = {}
for algo_name, algo_class in [('PPO', PPO), ('SAC', SAC), ('DDPG', DDPG), ('TD3', TD3)]:
    print(f"Training {algo_name} (NLP)...")
    agent = algo_class('MlpPolicy', nlp_env, verbose=1)
    agent.learn(total_timesteps=30000)
    base_agents_nlp[algo_name] = agent
```

#### 5. Train Meta-Agents

```python
class MetaAgentEnv(gym.Env):
    """Aggregates 4 base agents within one stream"""
    
    def __init__(self, base_agents: dict, features: np.ndarray):
        super().__init__()
        self.base_agents = base_agents
        self.features = features
        self.n_assets = 7
        
        # Observation: 4 agents × 7 weights + feature_dim
        # Data: 28 + 56 = 84-dim
        # NLP: 28 + 35 = 63-dim
        obs_dim = 4 * self.n_assets + features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )

# Train meta-agents
meta_data = PPO('MlpPolicy', meta_data_env, verbose=1)
meta_data.learn(total_timesteps=15000)

meta_nlp = PPO('MlpPolicy', meta_nlp_env, verbose=1)
meta_nlp.learn(total_timesteps=15000)
```

#### 6. Train Super-Agent

```python
class SuperAgentEnv(gym.Env):
    """Integrates both meta-agents"""
    
    def __init__(self, meta_data, meta_nlp):
        super().__init__()
        self.meta_data = meta_data
        self.meta_nlp = meta_nlp
        self.n_assets = 7
        
        # Observation: 7 (meta-data) + 7 (meta-nlp) + 1 (regime) = 15-dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets,), dtype=np.float32
        )

# Train final super-agent
super_agent = PPO('MlpPolicy', super_env, verbose=1)
super_agent.learn(total_timesteps=10000)
```

### Evaluation

```python
def evaluate_portfolio(agent, test_env, initial_capital=100000):
    """Evaluate agent performance on test period"""
    
    obs = test_env.reset()
    portfolio_values = [initial_capital]
    daily_returns = []
    allocations = []
    
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        
        portfolio_values.append(info['portfolio_value'])
        daily_returns.append(info['daily_return'])
        allocations.append(action)
    
    # Calculate metrics
    portfolio_values = np.array(portfolio_values)
    daily_returns = np.array(daily_returns)
    
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100
    annual_return = ((portfolio_values[-1] / portfolio_values[0]) ** (252/len(daily_returns)) - 1) * 100
    
    # Sharpe Ratio
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    
    # Volatility
    volatility = daily_returns.std() * np.sqrt(252) * 100
    
    # Maximum Drawdown
    cumulative = portfolio_values / np.maximum.accumulate(portfolio_values)
    max_drawdown = (cumulative.min() - 1) * 100
    
    results = {
        'total_roi': total_return,
        'annual_roi': annual_return,
        'sharpe_ratio': sharpe,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'final_value': portfolio_values[-1],
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'allocations': allocations
    }
    
    return results

# Evaluate all agents
print("Evaluating Super-Agent...")
super_results = evaluate_portfolio(super_agent, test_env)

print(f"Total ROI: {super_results['total_roi']:.2f}%")
print(f"Annualized ROI: {super_results['annual_roi']:.2f}%")
print(f"Sharpe Ratio: {super_results['sharpe_ratio']:.2f}")
print(f"Volatility: {super_results['volatility']:.2f}%")
print(f"Max Drawdown: {super_results['max_drawdown']:.2f}%")

# Compare with baselines
equal_weighted_results = evaluate_portfolio(equal_weighted_agent, test_env)
sp500_results = evaluate_portfolio(sp500_agent, test_env)

# Visualization
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(super_results['portfolio_values'], label='Super-Agent', linewidth=2)
plt.plot(equal_weighted_results['portfolio_values'], label='Equal-Weighted', alpha=0.7)
plt.plot(sp500_results['portfolio_values'], label='S&P 500 Only', alpha=0.7)
plt.xlabel('Trading Days')
plt.ylabel('Portfolio Value ($)')
plt.title('Cumulative Portfolio Performance')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

### Analyzing Portfolio Allocations

```python
def analyze_allocations(results, asset_names):
    """Analyze how the agent allocates across assets"""
    
    allocations = np.array(results['allocations'])
    
    # Monthly average allocations
    monthly_alloc = pd.DataFrame(
        allocations,
        columns=asset_names
    ).resample('M').mean()
    
    # Plot allocation over time
    fig, ax = plt.subplots(figsize=(14, 8))
    monthly_alloc.plot(kind='area', stacked=True, ax=ax, alpha=0.7)
    ax.set_ylabel('Allocation (%)')
    ax.set_xlabel('Date')
    ax.set_title('Portfolio Allocation Over Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Asset preference analysis
    avg_allocations = allocations.mean(axis=0)
    
    print("\nAverage Allocations:")
    for asset, alloc in zip(asset_names, avg_allocations):
        print(f"{asset:20s}: {alloc*100:6.2f}%")
    
    # Allocation volatility (how often does agent change strategy)
    alloc_changes = np.abs(np.diff(allocations, axis=0)).mean()
    print(f"\nAverage Daily Rebalancing: {alloc_changes*100:.2f}%")

# Run analysis
asset_names = ['S&P 500', 'NASDAQ', 'DOW', 'Russell', 'Gold', 'Silver', 'Oil']
analyze_allocations(super_results, asset_names)
```

### Quick Start with Pre-trained Models

```python
from src.models import load_pretrained_super_agent

# Load pre-trained super-agent
agent = load_pretrained_super_agent('models/super_agent/best_model.zip')

# Get portfolio allocation for current market conditions
allocation = agent.predict(current_observation)
print(allocation)  # [0.23, 0.15, 0.08, 0.02, 0.35, 0.12, 0.05]
```

---

## 🔬 Methodology

### Reinforcement Learning Algorithms

We train four different RL algorithms at the base tier:

1. **PPO (Proximal Policy Optimization)**
   - Stable on-policy learning
   - Clipped objective for conservative updates

2. **SAC (Soft Actor-Critic)**
   - Off-policy with entropy regularization
   - Encourages exploration

3. **DDPG (Deep Deterministic Policy Gradient)**
   - Deterministic policy for continuous actions
   - Experience replay buffer

4. **TD3 (Twin Delayed DDPG)**
   - Reduces value overestimation
   - Delayed policy updates

### Reward Function

Portfolio allocations are optimized using:

```python
R_t = r_t - 0.5 * σ_t - 2.0 * DD_t
```

Where:
- `r_t` = daily portfolio return
- `σ_t` = 30-day rolling volatility
- `DD_t` = drawdown from recent peak

This formulation balances returns with risk management.

### Training Details

- **Base Agents:** 30,000 timesteps each (~2 hours total on Google Colab)
- **Meta Agents:** 15,000 timesteps each
- **Super Agent:** 10,000 timesteps
- **Random Seed:** 42 (for reproducibility)
- **Environment:** Daily rebalancing, no transaction costs

---

## 📈 Results

### Cumulative Returns Over Time

The super-agent demonstrates consistent outperformance:

![Cumulative Returns](docs/images/cumulative_returns.png)

### Dynamic Allocation Strategy

Monthly allocation shifts show adaptive behavior:

| Month | S&P 500 | NASDAQ | Gold | Oil | Others |
|-------|---------|--------|------|-----|--------|
| Feb 2025 | 45.3% | 0.0% | 12.1% | 13.1% | 29.5% |
| Jun 2025 | 23.3% | 11.9% | 24.9% | 42.2% | -2.3% |
| Oct 2025 | 2.4% | 8.5% | 68.7% | 20.8% | -0.4% |

The agent dynamically shifts between equities and commodities based on market conditions.

---

## 📄 Research Paper

Full technical details available in our paper:

**"Reinforcement Learning for Portfolio Management"**  
*Aditya Chawla, Brian Zhang, Shyam Kannan*

[📄 Read Paper](docs/RL_Paper_WIP.pdf)

### Key Contributions

1. **Novel three-tier hierarchical architecture** separating algorithmic diversity from information fusion
2. **Dual-stream approach** combining market data with social sentiment
3. **Real-world validation** on actual historical data (not simulations)
4. **Superior risk-adjusted returns** with 2.89 Sharpe ratio

---

## 🤝 Contributors

This project was developed as part of advanced machine learning research:

| Name | Role | Contributions |
|------|------|---------------|
| **Shyam Kannan** | Data & Infrastructure | Data preprocessing, environment design, feature engineering |
| **Aditya Chawla** | Architecture & Training | Hierarchical RL system, agent training, model optimization |
| **Brian Zhang** | Evaluation & Analysis | Performance testing, metrics computation, visualization |

---

## 📚 References

### Key Papers

[1] Zhao & Welsch (2024) - "Hierarchical Reinforced Trader (HRT)"  
[2] Yang et al. (2020) - "FinBERT: Financial Language Model"  
[3] Koratamaddi et al. (2021) - "Market Sentiment-Aware Deep RL"

### Datasets

- **Market Data:** Yahoo Finance (2022-2025)
- **Sentiment Data:** Reddit (r/stocks, r/investing, r/wallstreetbets, r/StockMarket)

---

## 📊 Future Work

- [ ] Expand to global equities, bonds, and cryptocurrencies
- [ ] Add transaction cost modeling
- [ ] Multi-seed evaluation with statistical significance tests
- [ ] Incorporate Twitter/X and financial news sentiment
- [ ] Walk-forward validation with periodic retraining
- [ ] Attention mechanisms for interpretability
- [ ] Comparison with professional fund benchmarks

---

## 🙏 Acknowledgments

- **Stable-Baselines3** team for robust RL implementations
- **FinBERT** authors for the financial language model
- **Yahoo Finance** and **Reddit** for data access
- **Google Colab** for computational resources

---
