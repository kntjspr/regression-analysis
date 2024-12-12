import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class ObesityAnalysisDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Regression Analysis Dashboard")
        self.root.state('zoomed')
        
        # Create main frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Create top frame for controls
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(fill='x', padx=5, pady=5)
        
        # Add file selection button
        self.file_btn = ttk.Button(self.control_frame, text="Select Data File", command=self.load_data)
        self.file_btn.pack(side='left', padx=5)
        
        # Add variable selection frames
        self.var_frame = ttk.LabelFrame(self.control_frame, text="Variable Selection")
        self.var_frame.pack(side='left', padx=5, fill='x', expand=True)
        
        # Dependent variable selection
        self.dep_var_frame = ttk.Frame(self.var_frame)
        self.dep_var_frame.pack(side='left', padx=5)
        ttk.Label(self.dep_var_frame, text="Dependent Variable:").pack(side='left')
        self.dep_var = tk.StringVar()
        self.dep_var_cb = ttk.Combobox(self.dep_var_frame, textvariable=self.dep_var, state='readonly')
        self.dep_var_cb.pack(side='left', padx=5)
        
        # Independent variables selection
        self.indep_var_frame = ttk.Frame(self.var_frame)
        self.indep_var_frame.pack(side='left', padx=5)
        ttk.Label(self.indep_var_frame, text="Independent Variables:").pack(side='left')
        self.var_listbox = tk.Listbox(self.indep_var_frame, selectmode='multiple', height=3)
        self.var_listbox.pack(side='left', padx=5)
        
        # Add analyze button
        self.analyze_btn = ttk.Button(self.control_frame, text="Analyze", command=self.perform_analysis)
        self.analyze_btn.pack(side='left', padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs
        self.tab1 = ttk.Frame(self.notebook)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab4 = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab1, text='Correlation Analysis')
        self.notebook.add(self.tab2, text='Regression Diagnostics')
        self.notebook.add(self.tab3, text='Feature Analysis')
        self.notebook.add(self.tab4, text='Statistical Results')
        
        # Initialize data
        self.df = None
        self.model = None
        
    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Excel files", "*.xlsx *.xls"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if file_path:
            try:
                if file_path.endswith(('.xlsx', '.xls')):
                    self.df = pd.read_excel(file_path)
                else:
                    self.df = pd.read_csv(file_path)
                    
                # Update variable selection dropdowns
                self.update_variable_selections()
                messagebox.showinfo("Success", "Data loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading data: {str(e)}")
    
    def update_variable_selections(self):
        if self.df is not None:
            # Update dependent variable combobox
            self.dep_var_cb['values'] = list(self.df.columns)
            if 'Actual ObesityLvl' in self.df.columns:
                self.dep_var.set('Actual ObesityLvl')
            else:
                self.dep_var.set(self.df.columns[-1])
            
            # Update independent variables listbox
            self.var_listbox.delete(0, tk.END)
            for col in self.df.columns:
                if col != self.dep_var.get():
                    self.var_listbox.insert(tk.END, col)
            
            # Select default independent variables
            default_vars = ['Height', 'Weight', 'Age', 'OWFamHistory', 'FFoodBtwnMeals', 
                          'FPhysAct', 'FAlcohol', 'Is Bike_Transpo', 'Is Motorbike_Transpo']
            for i, item in enumerate(self.var_listbox.get(0, tk.END)):
                if item in default_vars:
                    self.var_listbox.selection_set(i)
    
    def perform_analysis(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        # Get selected variables
        dep_var = self.dep_var.get()
        indep_vars = [self.var_listbox.get(i) for i in self.var_listbox.curselection()]
        
        if not indep_vars:
            messagebox.showerror("Error", "Please select at least one independent variable!")
            return
            
        # Prepare data for analysis
        X = self.df[indep_vars]
        y = self.df[dep_var]
        
        # Clear previous plots
        for widget in self.tab1.winfo_children():
            widget.destroy()
        for widget in self.tab2.winfo_children():
            widget.destroy()
        for widget in self.tab3.winfo_children():
            widget.destroy()
        for widget in self.tab4.winfo_children():
            widget.destroy()
        
        # Correlation Analysis (Tab 1)
        fig1 = plt.Figure(figsize=(10, 8))
        ax1 = fig1.add_subplot(111)
        correlation_matrix = self.df[indep_vars + [dep_var]].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', ax=ax1)
        ax1.set_title('Correlation Heatmap')
        canvas1 = FigureCanvasTkAgg(fig1, self.tab1)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill='both', expand=True)
        
        # Regression Analysis
        X_with_const = sm.add_constant(X)
        self.model = sm.OLS(y, X_with_const).fit()
        
        # Regression Diagnostics (Tab 2)
        fig2 = plt.Figure(figsize=(12, 8))
        
        # Residual Plot
        ax2_1 = fig2.add_subplot(221)
        residuals = self.model.resid
        fitted_values = self.model.fittedvalues
        ax2_1.scatter(fitted_values, residuals)
        ax2_1.axhline(y=0, color='r', linestyle='--')
        ax2_1.set_xlabel('Fitted Values')
        ax2_1.set_ylabel('Residuals')
        ax2_1.set_title('Residual Plot')
        
        # Q-Q Plot
        ax2_2 = fig2.add_subplot(222)
        stats.probplot(residuals, dist="norm", plot=ax2_2)
        ax2_2.set_title('Normal Q-Q Plot')
        
        # Histogram of Residuals
        ax2_3 = fig2.add_subplot(223)
        ax2_3.hist(residuals, bins=20, edgecolor='black')
        ax2_3.set_xlabel('Residuals')
        ax2_3.set_ylabel('Frequency')
        ax2_3.set_title('Distribution of Residuals')
        
        # Actual vs Predicted
        ax2_4 = fig2.add_subplot(224)
        ax2_4.scatter(y, fitted_values)
        ax2_4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        ax2_4.set_xlabel(f'Actual {dep_var}')
        ax2_4.set_ylabel(f'Predicted {dep_var}')
        ax2_4.set_title('Actual vs Predicted Values')
        
        fig2.tight_layout()
        canvas2 = FigureCanvasTkAgg(fig2, self.tab2)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill='both', expand=True)
        
        # Feature Analysis (Tab 3)
        fig3 = plt.Figure(figsize=(12, 8))
        ax3 = fig3.add_subplot(111)
        
        coefficients = pd.DataFrame({
            'Feature': indep_vars,
            'Coefficient': self.model.params[1:],
            'Std Error': self.model.bse[1:]
        }).sort_values('Coefficient', ascending=True)
        
        ax3.barh(range(len(coefficients)), coefficients['Coefficient'])
        ax3.set_yticks(range(len(coefficients)))
        ax3.set_yticklabels(coefficients['Feature'])
        ax3.set_xlabel('Coefficient Value')
        ax3.set_title('Feature Importance in Predicting ' + dep_var)
        
        canvas3 = FigureCanvasTkAgg(fig3, self.tab3)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill='both', expand=True)
        
        # Statistical Results (Tab 4)
        text_widget = tk.Text(self.tab4, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill='both', expand=True)
        
        # Dataset Information
        text_widget.insert(tk.END, "DATASET INFORMATION:\n")
        text_widget.insert(tk.END, "-------------------\n")
        total_entries = len(self.df)
        valid_entries = len(y.dropna())  # Count non-null entries
        missing_entries = total_entries - valid_entries
        
        text_widget.insert(tk.END, f"Total Entries: {total_entries}\n")
        text_widget.insert(tk.END, f"Valid Entries Used: {valid_entries}\n")
        text_widget.insert(tk.END, f"Missing/Invalid Entries: {missing_entries}\n")
        text_widget.insert(tk.END, f"Coverage: {(valid_entries/total_entries)*100:.2f}%\n\n")
        
        # Variable Information
        text_widget.insert(tk.END, "VARIABLE INFORMATION:\n")
        text_widget.insert(tk.END, "--------------------\n")
        text_widget.insert(tk.END, f"Dependent Variable: {dep_var}\n")
        text_widget.insert(tk.END, f"Number of Independent Variables: {len(indep_vars)}\n")
        text_widget.insert(tk.END, "Independent Variables:\n")
        for var in indep_vars:
            missing = self.df[var].isna().sum()
            text_widget.insert(tk.END, f"  - {var}: {total_entries-missing} valid entries ({(total_entries-missing)/total_entries*100:.1f}% coverage)\n")
        text_widget.insert(tk.END, "\n")

        # Add statistical results to text widget
        text_widget.insert(tk.END, "REGRESSION ANALYSIS SUMMARY\n")
        text_widget.insert(tk.END, "==========================\n\n")
        
        # Key Statistics
        text_widget.insert(tk.END, "KEY METRICS:\n")
        text_widget.insert(tk.END, "-----------\n")
        text_widget.insert(tk.END, f"R-squared: {self.model.rsquared:.4f}\n")
        text_widget.insert(tk.END, f"Adjusted R-squared: {self.model.rsquared_adj:.4f}\n")
        text_widget.insert(tk.END, f"F-statistic: {self.model.fvalue:.4f}\n")
        text_widget.insert(tk.END, f"Prob (F-statistic): {self.model.f_pvalue:.4f}\n")
        text_widget.insert(tk.END, f"Standard Error of Regression: {np.sqrt(self.model.mse_resid):.4f}\n\n")

        # Error Metrics
        text_widget.insert(tk.END, "ERROR METRICS:\n")
        text_widget.insert(tk.END, "-------------\n")
        
        # Calculate various error metrics
        residuals = self.model.resid
        y_pred = self.model.fittedvalues
        y_true = y
        
        # MSE (Mean Squared Error)
        mse = np.mean(residuals**2)
        text_widget.insert(tk.END, f"MSE (Mean Squared Error): {mse:.4f}\n")
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mse)
        text_widget.insert(tk.END, f"RMSE (Root Mean Squared Error): {rmse:.4f}\n")
        
        # NRMSE (Normalized Root Mean Squared Error)
        y_range = y_true.max() - y_true.min()
        nrmse = rmse / y_range
        text_widget.insert(tk.END, f"NRMSE (Normalized RMSE): {nrmse:.4f}\n")
        
        # BSME (Balanced Mean Square Error)
        bias = np.mean(residuals)
        variance = np.var(residuals)
        bsme = bias**2 + variance
        text_widget.insert(tk.END, f"BSME (Balanced MSE): {bsme:.4f}\n")
        
        # Model MSE Improvement
        baseline_mse = np.mean((y_true - np.mean(y_true))**2)
        mse_improvement = (baseline_mse - mse) / baseline_mse * 100
        text_widget.insert(tk.END, f"Model MSE Improvement: {mse_improvement:.2f}%\n\n")

        # Additional Error Analysis
        text_widget.insert(tk.END, "ERROR DISTRIBUTION:\n")
        text_widget.insert(tk.END, "------------------\n")
        text_widget.insert(tk.END, f"Mean Error: {np.mean(residuals):.4f}\n")
        text_widget.insert(tk.END, f"Error Std Dev: {np.std(residuals):.4f}\n")
        text_widget.insert(tk.END, f"Max Error: {np.max(np.abs(residuals)):.4f}\n")
        text_widget.insert(tk.END, f"Median Absolute Error: {np.median(np.abs(residuals)):.4f}\n\n")

        # Coefficients and Confidence Intervals
        text_widget.insert(tk.END, "COEFFICIENTS AND CONFIDENCE INTERVALS (95%):\n")
        text_widget.insert(tk.END, "----------------------------------------\n")
        conf_int = self.model.conf_int()
        coef_summary = pd.DataFrame({
            'Coefficient': self.model.params,
            'Std Error': self.model.bse,
            'T-Value': self.model.tvalues,
            'P-Value': self.model.pvalues,
            'CI Lower': conf_int[0],
            'CI Upper': conf_int[1]
        })
        text_widget.insert(tk.END, str(coef_summary.round(4)) + "\n\n")

        # ANOVA Table
        text_widget.insert(tk.END, "ANOVA TABLE:\n")
        text_widget.insert(tk.END, "------------\n")
        
        # Calculate ANOVA components
        df_total = len(y) - 1
        df_residual = len(y) - len(indep_vars) - 1
        df_regression = len(indep_vars)

        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum(self.model.resid**2)
        ss_regression = ss_total - ss_residual

        ms_regression = ss_regression / df_regression
        ms_residual = ss_residual / df_residual

        f_stat = ms_regression / ms_residual
        p_value = 1 - stats.f.cdf(f_stat, df_regression, df_residual)

        anova_table = pd.DataFrame({
            'Source': ['Regression', 'Residual', 'Total'],
            'DF': [df_regression, df_residual, df_total],
            'SS': [ss_regression, ss_residual, ss_total],
            'MS': [ms_regression, ms_residual, np.nan],
            'F': [f_stat, np.nan, np.nan],
            'P-value': [p_value, np.nan, np.nan]
        })
        text_widget.insert(tk.END, str(anova_table.round(4)) + "\n\n")

        # Correlation Analysis
        text_widget.insert(tk.END, "CORRELATION MATRIX:\n")
        text_widget.insert(tk.END, "------------------\n")
        correlation_matrix = self.df[indep_vars + [dep_var]].corr().round(4)
        text_widget.insert(tk.END, str(correlation_matrix) + "\n\n")

        # Model Diagnostics
        text_widget.insert(tk.END, "MODEL DIAGNOSTICS:\n")
        text_widget.insert(tk.END, "-----------------\n")
        
        # Calculate diagnostic statistics
        dw_statistic = durbin_watson(self.model.resid)
        _, bp_pvalue, _, _ = het_breuschpagan(self.model.resid, self.model.model.exog)
        jb_statistic, jb_pvalue = stats.jarque_bera(self.model.resid)
        
        # Display diagnostic results
        text_widget.insert(tk.END, f"Durbin-Watson statistic: {dw_statistic:.4f}\n")
        text_widget.insert(tk.END, f"Breusch-Pagan test p-value: {bp_pvalue:.4f}\n")
        text_widget.insert(tk.END, f"Jarque-Bera test statistic: {jb_statistic:.4f}\n")
        text_widget.insert(tk.END, f"Jarque-Bera test p-value: {jb_pvalue:.4f}\n\n")

        # VIF Analysis
        text_widget.insert(tk.END, "VARIANCE INFLATION FACTORS:\n")
        text_widget.insert(tk.END, "--------------------------\n")
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
        text_widget.insert(tk.END, str(vif_data.round(4)))

        text_widget.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = ObesityAnalysisDashboard(root)
    root.mainloop()
