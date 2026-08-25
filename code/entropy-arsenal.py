import pybaseball as pb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import time
from sklearn.linear_model import LinearRegression
import plotly.express as px
from scipy.stats import skew, pearsonr, spearmanr, zscore

df = pb.statcast(start_dt = '2025-03-01', end_dt = '2025-11-01')
columns = ['pitcher', 'player_name', 'pitch_type', 'game_date', 'pfx_x', 'pfx_z']
df = df[columns]
df = df.dropna(subset = ['pfx_x', 'pfx_z'])
df['pfx_x'] = df['pfx_x'] * 12
df['pfx_z'] = df['pfx_z'] * 12

class Arsenal:
    """
    Represents a single pitcher's Statcast arsenal.
    
    Attributes
    ----------
    player_name : str
        Statcast player name ('Last, First').
    df : DataFrame
        All pitches thrown by the pitcher.
    name : str
        Player's name ('First, Last').
    movement : ndarray
        Horizontal and induced vertical break in inches.
    kde : gaussian_kde
        Kernel density estimate of movement profile.
    entropy : float
        Movement entropy (computed later).
    """
    GRID_SIZE = 75
    X_MAX = 30
    X_MIN = -30
    Y_MAX = 30
    Y_MIN = -30
    BANDWIDTH = 1000 ** (-1/16)
    SAMPLE_SIZE = 1000

    def __init__(self, df, player_name):
        self.player_name = player_name
        self.df = (df[df['player_name'] == player_name].copy())
        self.name = self.format_name()
        self.movement = self.df[['pfx_x', 'pfx_z']].astype(float).to_numpy()
        self.kde = None
        self.entropy = None
    
    def format_name(self):
        last, first = self.player_name.split(", ")
        return f'{first} {last}'
    
    # Plotting each pitcher's complete movement profile
    def plot_movement(self):
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.scatter(self.movement[:, 0], self.movement[:, 1], s = 5, alpha = 0.25)
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_xlabel('Horizontal Break (in.)')
        ax.xaxis.set_label_coords(0.5, -0.08)
        ax.set_ylabel('Induced Vertical Break (in.)')
        ax.yaxis.set_label_coords(-0.12, 0.5)
        ax.set_title(f'{self.name} Movement Profile')
        ax.spines['left'].set_position(('data', 0))
        ax.spines['right'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_color('none')
        ax.tick_params(axis='x', which='both', pad=8)
        ax.tick_params(axis='y', which='both', pad=8)
        ax.set_aspect('equal', adjustable = 'box')
        ax.invert_xaxis()
        ax.grid(alpha = 0.3)
        return fig, ax

    # Plotting each pitcher's complete movement profile, including a color-coded contouring for densities
    def plot_density(self):
        if self.kde is None:
            self.fit_density()
        self.levels = [self.Z.max()*0.05, self.Z.max()*0.10, self.Z.max()*0.25, self.Z.max()*0.50, self.Z.max()*0.75, self.Z.max()]
        fig, ax = self.plot_movement()
        ax.contour(self.X, self.Y, self.Z, levels = self.levels)
        ax.contourf(self.X, self.Y, self.Z, levels = self.levels, alpha = 0.25)
        plt.show()

    # Fitting and storing each pitcher's density values, along with the other metrics needed to calculate entropy
    def fit_density(self):
        self.kde, self.x, self.y, self.X, self.Y, self.Z = self._compute_density(self.movement)
    
    # Calculating and storing each pitcher's entropy value
    def calculate_entropy(self):
        if self.kde is None:
            self.fit_density()
        self.entropy = self._compute_entropy(self.x, self.y, self.Z)
    
    # Purely calculating density and other metrics for entropy, @staticmethod so can input any movement data
    @staticmethod
    def _compute_density(movement):
        values = movement.T
        kde = gaussian_kde(values, Arsenal.BANDWIDTH)
        x = np.linspace(Arsenal.X_MIN, Arsenal.X_MAX, Arsenal.GRID_SIZE)
        y = np.linspace(Arsenal.Y_MIN, Arsenal.Y_MAX, Arsenal.GRID_SIZE)
        X, Y = np.meshgrid(x, y)
        positions = np.vstack([X.ravel(), Y.ravel()])
        t0 = time.perf_counter()
        Z = kde(positions)
        Z = Z.reshape(X.shape)
        return kde, x, y, X, Y, Z
    
    # Purely calculating entropy, @staticmethod so can input any x, y, or Z values
    @staticmethod
    def _compute_entropy(x, y, Z):
        # Continuous entropy calculation
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        entropy = -np.sum(Z[Z > 0] * np.log(Z[Z > 0]) * dx * dy)
        return entropy
    
    # One function for calculating density and entropy for each pitcher
    def full_entropy_calculation(self):
        if Arsenal.SAMPLE_SIZE > len(self.df):
            return
        subset = self.df.sample(Arsenal.SAMPLE_SIZE)
        movement = subset[['pfx_x', 'pfx_z']].astype(float).to_numpy()
        kde, x, y, X, Y, Z = Arsenal._compute_density(movement)
        entropy = Arsenal._compute_entropy(x, y, Z)
        return entropy

    # Function used to analyze stability of entropy values with various sample sizes for pitch counts
    def stability_analysis(self):
        sample_sizes = [750, 800, 850, 900, 950, 1000, 1250]
        n_samples = 100
        results = []
        for sample_size in sample_sizes:
            if sample_size > len(self.df):
                break
            t0 = time.perf_counter()
            for _ in range(n_samples):
                subset = self.df.sample(sample_size)
                movement = subset[['pfx_x', 'pfx_z']].astype(float).to_numpy()
                kde, x, y, X, Y, Z = self._compute_density(movement)
                entropy = Arsenal._compute_entropy(x, y, Z)
                run_time = time.perf_counter() - t0
                results.append({'sample_size': sample_size, 'entropy': entropy, 'run_time': run_time})
        self.results = pd.DataFrame(results)
        self.summary = (self.results.groupby('sample_size').agg(mean_entropy = ('entropy', 'mean'), sd_entropy = ('entropy', 'std'), 
                                                                 mean_run_time = ('run_time', 'mean'), 
                                                                 sd_run_time = ('run_time', 'std')).reset_index())
        self.summary['cv_entropy'] = self.summary['sd_entropy'] / self.summary['mean_entropy']
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Entropy')
        ax.errorbar(self.summary['sample_size'], self.summary['mean_entropy'], 
                    yerr = self.summary['sd_entropy'], fmt = 'o-', capsize = 5)
        plt.show()
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.set_xlabel('Sample Size')
        ax.set_ylabel('Total Run Time')
        ax.errorbar(self.summary['sample_size'], self.summary['mean_run_time'], 
                    yerr = self.summary['sd_run_time'], fmt = 'o-', capsize = 5)
        plt.show()
        return pd.DataFrame(self.summary)
    
    # Same function as stability_analysis(self), but does not plot values
    def stability_no_plot(self):
        sample_sizes = [750, 800, 850, 900, 950, 1000, 1250]
        n_samples = 100
        results = []
        for sample_size in sample_sizes:
            if sample_size > len(self.df):
                break
            t0 = time.perf_counter()
            for _ in range(n_samples):
                subset = self.df.sample(sample_size)
                movement = subset[['pfx_x', 'pfx_z']].astype(float).to_numpy()
                kde, x, y, X, Y, Z = self._compute_density(movement)
                entropy = Arsenal._compute_entropy(x, y, Z)
                run_time = time.perf_counter() - t0
                results.append({'sample_size': sample_size, 'entropy': entropy, 'run_time': run_time})
        self.results = pd.DataFrame(results)
        self.summary = (self.results.groupby('sample_size').agg(mean_entropy = ('entropy', 'mean'), sd_entropy = ('entropy', 'std'), 
                                                                 mean_run_time = ('run_time', 'mean'), 
                                                                 sd_run_time = ('run_time', 'std')).reset_index())
        self.summary['cv_entropy'] = self.summary['sd_entropy'] / self.summary['mean_entropy']
        return pd.DataFrame(self.summary)

    # Function used to analyze stability of entropy values with various grid sizes for density calculation
    def stability_analysis2(self):
        sample_sizes = [75, 100, 125, 150, 175, 200, 250]
        n_samples = 100
        results2 = []
        for sample_size in sample_sizes:
            sample_start = time.perf_counter()
            for _ in range(n_samples):
                subset = self.df.sample(1000)
                movement = subset[['pfx_x', 'pfx_z']].astype(float).to_numpy()
                t0 = time.perf_counter()
                kde, x, y, X, Y, Z = self._compute_density(movement, sample_size)
                density_time = time.perf_counter() - t0
                entropy = Arsenal._compute_entropy(x, y, Z)
                total_time = time.perf_counter() - sample_start
                results2.append({'grid_size': sample_size, 'entropy': entropy, 'density_time': density_time, 'total_run_time': total_time})
        self.results2 = pd.DataFrame(results2)
        self.summary2 = (self.results2.groupby('sample_size').agg(mean_entropy = ('entropy', 'mean'), sd_entropy = ('entropy', 'std'), 
                                                                 mean_run_time = ('total_run_time', 'mean'), sd_run_time = ('run_time', 'std'),
                                                                 mean_density_time = ('density_time', 'mean'), 
                                                                 sd_density_time = ('density_time', 'std')).reset_index())
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.set_xlabel('Grid Size')
        ax.set_ylabel('Entropy')
        ax.errorbar(self.summary2['grid_size'], self.summary2['mean_entropy'], 
                    yerr = self.summary2['sd_entropy'], fmt = 'o-', capsize = 5)
        plt.show()
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.set_xlabel('Grid Size')
        ax.set_ylabel('Density Run Time')
        ax.errorbar(self.summary2['grid_size'], self.summary2['mean_run_time'], 
                    yerr = self.summary2['sd_run_time'], fmt = 'o-', capsize = 5)
        plt.show()
        return pd.DataFrame(self.summary2)
    
    # Function used to analyze stability of entropy values with various KDE bandwidths for density calculation
    def stability_analysis3(self):
        sample_bws = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 'scott']
        n_samples = 100
        results3 = []
        for sample_bw in sample_bws:
            for _ in range(n_samples):
                subset = self.df.sample(1000)
                movement = subset[['pfx_x', 'pfx_z']].astype(float).to_numpy()
                kde, x, y, X, Y, Z = self._compute_density(movement, Arsenal.GRID_SIZE, sample_bw)
                entropy = Arsenal._compute_entropy(x, y, Z)
                results3.append({'sample_bw': sample_bw, 'entropy': entropy})
        self.results3 = pd.DataFrame(results3)
        self.results3['bw_numeric'] = pd.to_numeric(self.results3['sample_bw'], errors = 'coerce')
        scott_rows = self.results3[self.results3['sample_bw'] == 'scott']
        scott_mean = scott_rows['entropy'].mean()
        self.summary3 = (self.results3.groupby('bw_numeric')
                         .agg(mean_entropy = ('entropy', 'mean'), sd_entropy = ('entropy', 'std')).reset_index())
        fig, ax = plt.subplots(figsize = (7, 7))
        ax.set_xlabel('Sample Bandwidth')
        ax.set_ylabel('Mean Entropy')
        ax.errorbar(self.summary3['bw_numeric'], self.summary3['mean_entropy'], 
                    yerr = self.summary3['sd_entropy'], fmt = 'o-', capsize = 5)
        ax.axhline(scott_mean, linestyle = '--', label = "Scott's Rule")
        ax.legend()
        plt.show()
        return pd.DataFrame(self.summary3)
                
# Initializing all pitchers within the dataset and assigning them their appropriate class
arsenals = {}

for name in df['player_name'].unique():
    arsenals[name] = Arsenal(df, name)

# Dataframe that contains every pitcher's total number of pitches recorded within the dataset
pitch_count = df.groupby('player_name').agg(total_pitches = ('player_name', 'count')).reset_index()

# How many pitchers have thrown at least 1000 pitches?
len(pitch_count[pitch_count['total_pitches'] >= 1000])

# Running stability analyses to test out various values for different variables (sample size pitch count, grid size, KDE bandwidth)
## Creating dataframe of 10 random pitchers that have thrown at least 1000 pitches
sample_pitchers = (pitch_count[pitch_count['total_pitches'] >= 1000].sample(10)['player_name'].tolist())
## results is a list that contains entropy values for each pitcher in sample_pitchers [(player_name): (entropy), etc.]
results = []
for pitcher in sample_pitchers:
    summary = arsenals[pitcher].stability_analysis()
    results.append(summary)
## Creating a DataFrame that puts every pitcher's list of values together and groups it by 
## the variable to be tested (in this case sample size pitch count)
sample_results = pd.concat(results, ignore_index = True)
sample_results = (sample_results.groupby('sample_size')
                  .agg(mean_cv_entropy = ('cv_entropy', 'mean'), sd_cv_entropy = ('cv_entropy', 'std')).reset_index())
## Plotting variable being tested (sample size pitch count again here) against the mean coefficient of variation to see at what 
## value for the variable the entropy begins to stabilize at (mean coefficient of variation changes minimally and levels out)
fig, ax = plt.subplots(figsize = (7, 7))
ax.set_xlabel('Sample Size (# of Pitches)')
ax.set_ylabel('Mean Coefficient of Variation')
ax.errorbar(sample_results['sample_size'], sample_results['mean_cv_entropy'], 
            yerr = sample_results['sd_cv_entropy'], fmt = 'o-', capsize = 5)
plt.show()

# Calculating entropy for the pitchers that have thrown at least the sample size (1000) number of pitches and 
# store it in the final dataframe alongside the performance variables to run correlatiion against
qual_pitchers_1000 = (pitch_count[pitch_count['total_pitches'] >= 1000]['player_name'].tolist())
## import dataframe with stats [using K%, xwOBA, FIP, HR/9, BB%, wOBA, and xSLG]
stats_file_dir = 'data/2025_performance_data.csv'
final_df = pd.read_csv(stats_file_dir)
fip_constant_2025 = 3.135
HR = final_df['hr']
BB = final_df['bb']
HBP = final_df['hbp']
K = final_df['k']
IP = final_df['IP']
final_df['FIP'] = (((13 * HR) + (3 * (BB + HBP)) - (2 * K)) / IP) + fip_constant_2025
final_df['HR/9'] = (HR / IP) * 9
stat_columns = ['K%', 'xwOBA', 'FIP', 'HR/9', 'BB%', 'wOBA', 'xSLG']
final_df = final_df[['player_name'] + ['IP'] + stat_columns]
final_df = final_df[final_df['player_name'].isin(qual_pitchers_1000)]
pitch_type_count = df.groupby('player_name').agg(arsenal_count = ('pitch_type', 'nunique'))
final_df = final_df.merge(pitch_type_count, on = 'player_name')

## calculate entropy and store it in complete, final DataFrame
entropy_results = []

for pitcher in qual_pitchers_1000:
    arsenal = Arsenal(df, pitcher)
    entropy = arsenal.full_entropy_calculation()
    entropy_results.append({'player_name': pitcher, 'entropy': entropy})

entropy_df = pd.DataFrame(entropy_results)
final_df = final_df.merge(entropy_df, on = 'player_name')
final_df['entropy/arsenal'] = final_df['entropy'] / final_df['arsenal_count']

# Calculating Pearson's r and p-value and plotting for each predictor and each performance metric
r_p_values = []
entropy_columns = ['entropy', 'arsenal_count', 'entropy/arsenal']

for entropy in entropy_columns:
    for stat in stat_columns:
        safe_entropy = entropy.replace('/', '_')
        safe_stat = stat.replace('/', '_')

        lr = LinearRegression(fit_intercept = True)
        lr.fit(final_df[[entropy]], final_df[stat])
        stat_pred = lr.predict(final_df[[entropy]])
        slope = lr.coef_[0]
        intercept = lr.intercept_
        pearson_r, pearson_p = pearsonr(final_df[entropy], final_df[stat])
        regression_info = f"y = {slope:.2f}x + {intercept:.2f}\nR² = {(pearson_r ** 2):.3f}"
        
        r_p_values.append({'Predictor': entropy, 'Outcome': stat, "Pearson's r": pearson_r, "Pearson's p-value": pearson_p})

        plot_final_df = final_df.sort_values(entropy)
        
        fig, ax = plt.subplots(figsize = (7, 5))
        ax.scatter(plot_final_df[entropy], plot_final_df[stat], color = 'steelblue', alpha = 0.6, s = 20)
        ax.plot(plot_final_df[entropy], stat_pred[plot_final_df.index], color = 'crimson', linewidth = 2)
        ax.set_title(f'Relationship Between {entropy} and {stat} in 2025')
        ax.set_xlabel(entropy)
        ax.set_ylabel(stat)
        ax.text(plot_final_df[entropy].min(), plot_final_df[stat].max(), regression_info, fontsize = 10, 
                verticalalignment = 'top', bbox = dict(facecolor = 'white', edgecolor = 'black', pad = 4))
        plt.tight_layout()
        plt.savefig(f"{safe_entropy}_plots/{safe_stat}.png", dpi = 300, bbox_inches = 'tight')
        plt.close(fig)

r_p_values = pd.DataFrame(r_p_values).sort_values("Pearson's p-value")
r_p_values["Pearson's r"] = r_p_values["Pearson's r"].round(3)
r_p_values["Pearson's p-value"] = r_p_values["Pearson's p-value"].round(3)
entropy_r_p_values = r_p_values[r_p_values['Predictor'] == 'entropy']
arsenal_count_r_p_values = r_p_values[r_p_values['Predictor'] == 'arsenal_count']
entropy_arsenal_r_p_values = r_p_values[r_p_values['Predictor'] == 'entropy/arsenal']

entropy_r_p_values.to_excel('entropy_table.xlsx', index = False)
arsenal_count_r_p_values.to_excel('arsenal_count_table.xlsx', index = False)
entropy_arsenal_r_p_values.to_excel('entropy_arsenal_table.xlsx', index = False)

entropy_arsenal_r, entropy_arsenal_p = pearsonr(final_df['entropy'], final_df['arsenal_count'])
print(f'Entropy vs. Arsenal Depth r: {entropy_arsenal_r}\nEntropy vs. Arsenal Depth p-value: {entropy_arsenal_p}')

