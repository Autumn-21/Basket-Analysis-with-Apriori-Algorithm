# Basket Analysis with Apriori Algorithm

## Project Description
This project implements market basket analysis using the Apriori algorithm to uncover product association patterns in retail transaction data. The analysis identifies frequently co-purchased items and generates actionable business insights to optimize sales strategies.

## Key Features
- **Association Rule Mining**: Implements Apriori algorithm to discover frequent item sets
- **Business Insights**: Identifies strong product associations (e.g., coffee + pastry)
- **Time-based Analysis**: Examines purchasing patterns by time of day and weekday/weekend
- **Visual Analytics**: Includes transaction frequency charts and item association graphs

## Technical Implementation
### Data Source
Bakery Sales Dataset from Kaggle containing:
- Transaction IDs
- Item lists
- Timestamps
- Daypart classifications (morning/afternoon/evening/night)
- Weekday/weekend indicators

### Tools & Libraries
- **Python 3**
- Jupyter Notebook environment
- Primary libraries:
  - `mlxtend` (Apriori implementation)
  - `pandas` (data manipulation)
  - `matplotlib`/`seaborn` (visualization)
  - `scikit-learn` (data preprocessing)

### Methodology
1. **Data Preprocessing**:
   - Handling missing values
   - Transaction transformation
   - One-hot encoding

2. **Exploratory Analysis**:
   - Transaction frequency by time
   - Item popularity analysis

3. **Association Rule Mining**:
   - Frequent itemset generation
   - Rule extraction (support/confidence/lift metrics)
   - Business interpretation

## Getting Started
1. Clone repository:
   ```bash
   git clone [repository-url]
   ```
2. Run Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
## Results & Applications
Key findings enable:

  - Strategic product bundling
  - Optimized inventory management
  - Time-targeted promotions
  - Store layout improvements
