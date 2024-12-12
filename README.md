# Regression Analysis Dashboard

An interactive Python application for comprehensive regression analysis with a user-friendly GUI interface.

## Features

### Data Handling
- Support for Excel (.xlsx, .xls) and CSV files
- Dynamic variable selection
- Automatic detection of dependent and independent variables
- Flexible dataset compatibility

### Statistical Analysis
- Multiple Linear Regression
- Correlation Analysis
- ANOVA
- Confidence Intervals
- Comprehensive Error Metrics

### Visualizations
1. **Correlation Analysis**
   - Interactive correlation heatmap
   - Visual representation of variable relationships

2. **Regression Diagnostics**
   - Residual plots
   - Q-Q plots for normality assessment
   - Residual distribution histogram
   - Actual vs Predicted values scatter plot

3. **Feature Analysis**
   - Feature importance visualization
   - Coefficient plots

### Statistical Metrics
- R-squared and Adjusted R-squared
- Standard Errors
- Coefficients with confidence intervals
- ANOVA table
- Error Metrics:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - NRMSE (Normalized RMSE)
  - BSME (Balanced MSE)
  - Model MSE Improvement

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kntjspr/regression-analysis-dashboard.git
cd regression-analysis-dashboard
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels tkinter openpyxl
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Using the Dashboard:
   - Click "Select Data File" to load your dataset
   - Choose dependent variable from dropdown
   - Select independent variables from listbox (use Ctrl+click for multiple)
   - Click "Analyze" to run the regression analysis

3. Viewing Results:
   - Navigate through tabs to view different aspects of analysis
   - All visualizations are interactive
   - Statistical results are displayed in organized sections

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- tkinter

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Python's scientific computing stack
- Uses statsmodels for statistical computations
- Visualization powered by matplotlib and seaborn 
