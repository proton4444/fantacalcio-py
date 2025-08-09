# Mantra Fantacalcio Strategy Engine 🏆

Enhanced Fantacalcio analysis tool specifically designed for the **Side Event Mantra 25/26** season with advanced module-based strategy, tier analysis, and auction optimization.

## 🚀 Quick Start

### Generate Your First Auction List

```bash
# Quick auction list with defaults (€500 budget, 8 teams, S+A tiers)
python auction_list.py --quick

# Custom auction list
python auction_list.py --budget 750 --teams 10 --tiers S,A,B --max-players 30

# Focus on specific role (e.g., only defenders)
python auction_list.py --role D --tiers S,A

# Analyze a specific player
python auction_list.py --player "Vlahovic"
```

## 📋 Features

- **🎯 Module-Based Analysis**: Optimized for 4-2-3-1 (primary), 4-3-3, and 4-4-2 formations
- **📊 Tier System**: S/A/B/C player classification based on Quotazione quantiles
- **🔄 Polyvalence Bonuses**: +10-15% value for players fitting multiple roles
- **💰 Budget Allocation**: Smart budget distribution across roles
- **🎲 Anti-Parity Bidding**: +1 strategy on round numbers
- **📈 Scarcity Analysis**: Role availability vs. module requirements
- **📱 CLI Tools**: Quick auction list generation

## 🛠️ Installation

```bash
# Clone your repository
git clone https://github.com/proton4444/fantacalcio-py.git
cd fantacalcio-py

# Switch to enhanced branch
git checkout mantra-strategy-enhancement

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage Guide

### 1. Data Collection

First, collect fresh player data from fantcalciopedia.com:

```bash
python mantra_strategy.py --collect-data
```

### 2. Quick Auction List

Generate an optimized auction list for your upcoming auction:

```bash
# Basic usage
python auction_list.py --budget 500 --teams 8

# Advanced usage
python auction_list.py \
  --budget 750 \
  --teams 10 \
  --tiers S,A,B \
  --max-players 40 \
  --role C
```

### 3. Full Strategy Analysis

Run comprehensive analysis with detailed reports:

```bash
python mantra_strategy.py --analyze
```

This generates:
- `enhanced_player_data.csv` - Complete player database with tiers and bonuses
- `auction_target_list.csv` - Prioritized auction targets
- `module_scarcity_analysis.csv` - Role scarcity analysis
- `budget_allocation.csv` - Recommended budget distribution

### 4. Player Analysis

Get detailed analysis for specific players:

```bash
python auction_list.py --player "Lautaro"
python auction_list.py --player "Barella"
```

## ⚙️ Configuration

Edit `config.yaml` to customize your strategy:

```yaml
# Update these for your specific auction
auction:
  total_budget: 500    # Your budget
  teams_count: 8       # Number of teams
  
# Adjust module requirements if needed
modules:
  primary:
    name: "4-2-3-1"
    roles:
      P: 3  # Portieri
      D: 8  # Difensori
      C: 8  # Centrocampisti
      T: 6  # Trequartisti
      A: 3  # Attaccanti
```

## 📊 Module Strategies

### Primary: 4-2-3-1
- **Strengths**: Balanced attack/defense, flexible midfield
- **Key Roles**: Strong trequartista, defensive midfielders
- **Budget Focus**: 30% defenders, 25% midfielders, 20% trequartisti

### Backup: 4-3-3
- **Strengths**: Wide attack, midfield control
- **Polyvalence**: Trequartisti as wingers, midfielders as wide players

### Backup: 4-4-2
- **Strengths**: Solid defense, direct attack
- **Polyvalence**: Trequartisti as wide midfielders

## 🎯 Tier System

- **S Tier (Top 10%)**: Premium players, highest quotazioni
- **A Tier (Next 25%)**: Quality starters, good value
- **B Tier (Next 40%)**: Solid options, budget-friendly
- **C Tier (Bottom 25%)**: Bench players, emergency picks

## 💡 Bidding Strategy

The tool applies these intelligent rules:

1. **Anti-Parity Rule**: +1 on round numbers (20→21, 15→16)
2. **Polyvalence Bonus**: +10-15% for multi-role players
3. **Scarcity Premium**: Higher bids for scarce roles
4. **Tier Multipliers**: S=+20%, A=+10%, B=0%, C=-10%

## 📈 Example Output

```
🎯 MANTRA AUCTION LIST GENERATED
============================================================
📊 Total Players: 25
💰 Budget: €500
👥 Teams: 8
🏆 Tier Focus: S, A
📁 Files saved to: reports/2025-08-09

🔥 TOP 10 TARGETS:
------------------------------------------------------------
 1. Lautaro Martinez    (A  ) Tier S - Bid: €51 +12%
 2. Barella            (C  ) Tier S - Bid: €43 +8%
 3. Vlahovic          (A  ) Tier S - Bid: €41
 4. Leao              (T  ) Tier A - Bid: €38 +15%
 5. Bastoni           (D  ) Tier A - Bid: €32
```

## 🔧 Advanced Features

### Custom Role Mappings

Modify polyvalence mappings in `mantra_strategy.py`:

```python
polyvalent_mappings = {
    'Centrocampisti': ['T'],      # Can play attacking mid
    'Trequartisti': ['C', 'A'],   # Can play central mid or forward
    'Difensori': ['C'],           # Can play defensive mid
}
```

### Budget Allocation Tuning

Adjust role budgets in `config.yaml`:

```yaml
budget_allocation:
  P: 0.15   # 15% for goalkeepers
  D: 0.30   # 30% for defenders
  C: 0.25   # 25% for midfielders
  T: 0.20   # 20% for attacking midfielders
  A: 0.10   # 10% for forwards
```

## 📁 File Structure

```
fantacalcio-py/
├── config.yaml              # Strategy configuration
├── mantra_strategy.py        # Main strategy engine
├── auction_list.py          # CLI auction list generator
├── requirements.txt         # Dependencies
├── reports/                 # Generated reports
│   └── YYYY-MM-DD/
│       ├── auction_target_list.csv
│       ├── budget_allocation.csv
│       └── enhanced_player_data.csv
└── v1/                     # Legacy files
```

## 🤖 Automation

### Weekly Data Updates

Set up automatic data collection:

```bash
# Add to crontab for weekly updates
0 9 * * 1 cd /path/to/fantacalcio-py && python mantra_strategy.py --collect-data
```

### Pre-Auction Checklist

1. Update player data: `python mantra_strategy.py --collect-data`
2. Verify config: Check `config.yaml` budget and teams
3. Generate list: `python auction_list.py --budget YOUR_BUDGET`
4. Review targets: Check top 20 players and budget allocation
5. Backup plan: Generate alternative lists for different tiers

## 🆘 Troubleshooting

### Data Collection Issues
```bash
# If scraping fails, check internet connection and try again
python mantra_strategy.py --collect-data

# For partial failures, manually delete giocatori.csv and retry
rm giocatori.csv giocatori_urls.txt
python mantra_strategy.py --collect-data
```

### Missing Dependencies
```bash
pip install -r requirements.txt
# or individual packages:
pip install pandas numpy matplotlib seaborn pyyaml
```

### Configuration Errors
- Verify `config.yaml` syntax (use YAML validator)
- Check role totals add up correctly
- Ensure budget allocations sum to 1.0 (100%)

## 📞 Support

For issues or enhancements:
1. Check existing GitHub issues
2. Create new issue with detailed description
3. Include your `config.yaml` and error messages

## 🎉 Good Luck!

May your auction be successful and your Mantra team dominate the Side Event 25/26! 🏆

---

*Made with ❤️ for the Mantra Fantacalcio community*