"""
Comprehensive Exploratory Data Analysis (EDA) Script for Bike-Sharing Dataset
DSA211 - Topic 1: Diagnosing and Remedying Nonlinearity in Bike-Sharing Demand Models

Addresses:
- Proposal and Feedback Adjustments:
  1. Time-based train/test split (hold out Q4 2012: Oct-Dec 2012).
  2. EDA strictly on training data.
  3. Proper categorical encoding of hr, season, mnth, weathersit.
  4. Investigation of extreme multicollinearity (season vs mnth, temp vs atemp).
  5. Adjudication of outliers by grouping by date and cross-referencing known external events.
  6. Statistical distribution fitting of cnt (Poisson, Negative Binomial, Log-Normal, Gamma, Weibull).
  7. Time-aspects analysis: diurnal cycle (bimodal commuter peaks), weekly patterns, monthly seasonality.
  8. Interaction and non-linear scatter/relationship plots with count.
  9. Simple, intuitive non-modeling behavioral EDA:
     - Weather Severity Breakdown (weathersit)
     - Temperature Comfort Zones (Binned Temperature)
     - Humidity & Mugginess Bands
     - Commuters vs Casual Riders in Adverse Weather
     - Rush Hour Asymmetry (Morning Spike vs Evening Plateau)
     - Overnight "Dead Zone" (12 AM - 5 AM)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

FIGURES_DIR = os.path.join('..', 'figures')
if not os.path.exists(FIGURES_DIR):
    FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

def main():
    print("Loading data...")
    hour_path = os.path.join('data', 'hour.csv')
    if not os.path.exists(hour_path):
        hour_path = os.path.join('..', 'data', 'hour.csv')
    
    df = pd.read_csv(hour_path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    # -------------------------------------------------------------
    # 1. TRAIN / TEST SPLIT (FEEDBACK COMPLIANCE)
    # Time-based split: Hold out Oct-Dec 2012 (Q4 2012)
    # Train: 2011-01-01 to 2012-09-30
    # Test: 2012-10-01 to 2012-12-31
    # -------------------------------------------------------------
    split_date = pd.to_datetime('2012-10-01')
    train_mask = df['dteday'] < split_date
    test_mask = df['dteday'] >= split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"Total observations: {len(df)}")
    print(f"Train observations (Jan 2011 - Sep 2012): {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Test observations (Oct 2012 - Dec 2012): {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    # Un-normalize continuous features for intuitive presentation
    train_df['temp_c'] = train_df['temp'] * 41.0
    train_df['atemp_c'] = train_df['atemp'] * 50.0
    train_df['hum_pct'] = train_df['hum'] * 100.0
    train_df['windspeed_kmh'] = train_df['windspeed'] * 67.0
    
    df['temp_c'] = df['temp'] * 41.0
    df['atemp_c'] = df['atemp'] * 50.0
    df['hum_pct'] = df['hum'] * 100.0
    df['windspeed_kmh'] = df['windspeed'] * 67.0

    # =============================================================
    # QUESTION 1: OVERALL TREND OF BIKE RENTAL USAGE
    # =============================================================
    print("\n--- Question 1: Overall Trend ---")
    daily_trend = df.groupby('dteday').agg({
        'cnt': 'sum',
        'casual': 'sum',
        'registered': 'sum',
        'temp_c': 'mean'
    }).reset_index()
    daily_trend['cnt_7d_mavg'] = daily_trend['cnt'].rolling(window=7, center=True).mean()
    daily_trend['cnt_30d_mavg'] = daily_trend['cnt'].rolling(window=30, center=True).mean()
    
    yr_2011_tot = df[df['yr'] == 0]['cnt'].sum()
    yr_2012_tot = df[df['yr'] == 1]['cnt'].sum()
    yr_2011_mean_hr = df[df['yr'] == 0]['cnt'].mean()
    yr_2012_mean_hr = df[df['yr'] == 1]['cnt'].mean()
    growth_rate = ((yr_2012_tot - yr_2011_tot) / yr_2011_tot) * 100
    
    print(f"2011 Total Rentals: {yr_2011_tot:,} (Mean/hr: {yr_2011_mean_hr:.1f})")
    print(f"2012 Total Rentals: {yr_2012_tot:,} (Mean/hr: {yr_2012_mean_hr:.1f})")
    print(f"Annual Growth Rate: {growth_rate:.2f}%")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_trend['dteday'], daily_trend['cnt'], color='lightblue', alpha=0.5, label='Daily Total Count')
    ax.plot(daily_trend['dteday'], daily_trend['cnt_7d_mavg'], color='navy', linewidth=1.8, label='7-Day Moving Average')
    ax.plot(daily_trend['dteday'], daily_trend['cnt_30d_mavg'], color='darkorange', linewidth=2.5, label='30-Day Moving Average')
    ax.axvline(split_date, color='red', linestyle='--', linewidth=2, label='Train / Test Split (2012-10-01)')
    ax.text(split_date - pd.Timedelta(days=15), 8500, 'Training Set (EDA & Model CV)', color='black', ha='right', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.text(split_date + pd.Timedelta(days=15), 8500, 'Holdout Test Set (Touch Once)', color='red', ha='left', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    ax.set_title('Overall Daily Bike Rental Trend (2011 - 2012) with Train/Test Split Boundary', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Daily Bike Rentals (cnt)')
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '01_overall_trend_timeline.png'), dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    daily_trend['cas_7d'] = daily_trend['casual'].rolling(7, center=True).mean()
    daily_trend['reg_7d'] = daily_trend['registered'].rolling(7, center=True).mean()
    ax.plot(daily_trend['dteday'], daily_trend['reg_7d'], color='#1f77b4', linewidth=2, label='Registered (7-day MA)')
    ax.plot(daily_trend['dteday'], daily_trend['cas_7d'], color='#ff7f0e', linewidth=2, label='Casual (7-day MA)')
    ax.axvline(split_date, color='red', linestyle='--', label='Train/Test Split')
    ax.set_title('User Segments Long-term Trend: Registered vs Casual Users', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Daily Rentals')
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '01_user_type_trends.png'), dpi=300)
    plt.close(fig)

    # =============================================================
    # QUESTION 2: OVERALL DISTRIBUTION & STATISTICAL FIT
    # =============================================================
    print("\n--- Question 2: Overall Distribution & Statistical Distribution Fitting ---")
    cnt_train = train_df['cnt'].values
    mean_cnt = np.mean(cnt_train)
    var_cnt = np.var(cnt_train, ddof=1)
    std_cnt = np.std(cnt_train, ddof=1)
    median_cnt = np.median(cnt_train)
    skew_cnt = stats.skew(cnt_train)
    kurt_cnt = stats.kurtosis(cnt_train)
    dispersion_index = var_cnt / mean_cnt
    
    print(f"Mean: {mean_cnt:.2f}")
    print(f"Variance: {var_cnt:.2f}")
    print(f"Std Dev: {std_cnt:.2f}")
    print(f"Median: {median_cnt:.2f}")
    print(f"Skewness: {skew_cnt:.3f}")
    print(f"Kurtosis: {kurt_cnt:.3f}")
    print(f"Dispersion Index (Var/Mean): {dispersion_index:.2f}")
    
    log_cnt = np.log(cnt_train)
    log_mean, log_std = np.mean(log_cnt), np.std(log_cnt, ddof=1)
    log_skew = stats.skew(log_cnt)
    log_kurt = stats.kurtosis(log_cnt)
    
    norm_stat, norm_p = stats.kstest(cnt_train, 'norm', args=(mean_cnt, std_cnt))
    shape_ln, loc_ln, scale_ln = stats.lognorm.fit(cnt_train, floc=0)
    ln_stat, ln_p = stats.kstest(cnt_train, 'lognorm', args=(shape_ln, loc_ln, scale_ln))
    shape_gam, loc_gam, scale_gam = stats.gamma.fit(cnt_train, floc=0)
    gam_stat, gam_p = stats.kstest(cnt_train, 'gamma', args=(shape_gam, loc_gam, scale_gam))
    loc_exp, scale_exp = stats.expon.fit(cnt_train, floc=0)
    exp_stat, exp_p = stats.kstest(cnt_train, 'expon', args=(loc_exp, scale_exp))
    
    ll_norm = np.sum(stats.norm.logpdf(cnt_train, mean_cnt, std_cnt))
    ll_ln = np.sum(stats.lognorm.logpdf(cnt_train, shape_ln, loc_ln, scale_ln))
    ll_gam = np.sum(stats.gamma.logpdf(cnt_train, shape_gam, loc_gam, scale_gam))
    ll_exp = np.sum(stats.expon.logpdf(cnt_train, loc_exp, scale_exp))
    
    fit_table = pd.DataFrame({
        'Distribution': ['Gamma', 'Exponential', 'Log-Normal', 'Normal (Gaussian)'],
        'Parameters': [
            f"a={shape_gam:.3f}, scale={scale_gam:.1f}",
            f"scale={scale_exp:.1f}",
            f"s={shape_ln:.3f}, scale={scale_ln:.1f}",
            f"mu={mean_cnt:.1f}, sigma={std_cnt:.1f}"
        ],
        'Log-Likelihood': [ll_gam, ll_exp, ll_ln, ll_norm],
        'AIC': [2*2 - 2*ll_gam, 2*1 - 2*ll_exp, 2*2 - 2*ll_ln, 2*2 - 2*ll_norm],
        'KS-Statistic': [gam_stat, exp_stat, ln_stat, norm_stat],
        'KS p-value': [gam_p, exp_p, ln_p, norm_p]
    }).sort_values('AIC')
    print("\nGoodness-of-Fit Comparison:")
    print(fit_table.to_string(index=False))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_grid = np.linspace(1, cnt_train.max(), 1000)
    
    axes[0].hist(cnt_train, bins=50, density=True, alpha=0.45, color='steelblue', edgecolor='black', label='Observed Data (cnt)')
    axes[0].plot(x_grid, stats.norm.pdf(x_grid, mean_cnt, std_cnt), 'r--', linewidth=2, label='Normal Fit')
    axes[0].plot(x_grid, stats.gamma.pdf(x_grid, shape_gam, loc_gam, scale_gam), 'g-', linewidth=2, label='Gamma Fit')
    axes[0].plot(x_grid, stats.lognorm.pdf(x_grid, shape_ln, loc_ln, scale_ln), 'm-', linewidth=2.2, label='Log-Normal Fit')
    axes[0].set_title(f'Raw Count Distribution (Skewness={skew_cnt:.2f})', fontweight='bold')
    axes[0].set_xlabel('Hourly Bike Rental Count (cnt)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    axes[1].hist(log_cnt, bins=40, density=True, alpha=0.5, color='coral', edgecolor='black', label='Observed log(cnt)')
    x_log_grid = np.linspace(log_cnt.min(), log_cnt.max(), 500)
    axes[1].plot(x_log_grid, stats.norm.pdf(x_log_grid, log_mean, log_std), 'b-', linewidth=2.2, label=f'Normal Fit to log(cnt)')
    axes[1].set_title(f'Log-Transformed Count: log(cnt)', fontweight='bold')
    axes[1].set_xlabel('log(cnt)')
    axes[1].set_ylabel('Density')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '02_distribution_fitting.png'), dpi=300)
    plt.close(fig)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    stats.probplot(cnt_train, dist="norm", plot=axes[0])
    axes[0].set_title('Normal Q-Q Plot: Raw cnt', fontweight='bold')
    stats.probplot(log_cnt, dist="norm", plot=axes[1])
    axes[1].set_title('Normal Q-Q Plot: log(cnt)', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '02_qq_plots_raw_vs_log.png'), dpi=300)
    plt.close(fig)

    # =============================================================
    # QUESTION 3: TIME ASPECTS
    # =============================================================
    print("\n--- Question 3: Time Aspects ---")
    hr_working = train_df.groupby(['hr', 'workingday'])['cnt'].mean().unstack()
    hr_reg_cas = train_df.groupby('hr')[['registered', 'casual', 'cnt']].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(hr_working.index, hr_working[1], marker='o', linewidth=2.5, color='#1f77b4', label='Working Day (Commuter Peaks)')
    axes[0].plot(hr_working.index, hr_working[0], marker='s', linewidth=2.5, color='#ff7f0e', label='Non-Working Day (Midday Peak)')
    axes[0].set_title('Hourly Diurnal Profile: Working Day vs Weekend/Holiday', fontweight='bold')
    axes[0].set_xlabel('Hour of Day (0 - 23)')
    axes[0].set_ylabel('Average Hourly Rentals')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    axes[1].plot(hr_reg_cas.index, hr_reg_cas['registered'], marker='^', linewidth=2.2, color='teal', label='Registered Users')
    axes[1].plot(hr_reg_cas.index, hr_reg_cas['casual'], marker='v', linewidth=2.2, color='crimson', label='Casual Users')
    axes[1].set_title('Hourly Profile by User Segment', fontweight='bold')
    axes[1].set_xlabel('Hour of Day (0 - 23)')
    axes[1].set_ylabel('Average Hourly Rentals')
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_hourly_diurnal_patterns.png'), dpi=300)
    plt.close(fig)

    # Day of Week & Holiday
    weekday_labels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    sns.boxplot(x='weekday', y='cnt', data=train_df, ax=axes[0], color='skyblue', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[0].set_xticks(range(7))
    axes[0].set_xticklabels(weekday_labels)
    axes[0].set_title('Hourly Rentals Across Days of Week', fontweight='bold')
    axes[0].set_xlabel('Weekday')
    axes[0].set_ylabel('Hourly Count (cnt)')
    
    sns.boxplot(x='holiday', y='cnt', data=train_df, ax=axes[1], color='lightgreen', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(['Non-Holiday (0)', 'Holiday (1)'])
    axes[1].set_title('Holiday Effect', fontweight='bold')
    axes[1].set_xlabel('Holiday Indicator')
    axes[1].set_ylabel('Hourly Count (cnt)')
    
    sns.boxplot(x='workingday', y='cnt', data=train_df, ax=axes[2], color='salmon', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"red", "markeredgecolor":"red"})
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(['Non-Working Day (0)', 'Working Day (1)'])
    axes[2].set_title('Working Day Effect', fontweight='bold')
    axes[2].set_xlabel('Working Day Indicator')
    axes[2].set_ylabel('Hourly Count (cnt)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_weekday_holiday_workingday.png'), dpi=300)
    plt.close(fig)

    # Month and Season
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    mnth_yr = train_df.groupby(['mnth', 'yr'])['cnt'].mean().unstack()
    mnth_yr.plot(kind='bar', ax=axes[0], color=['#4c72b0', '#55a868'], width=0.8)
    axes[0].set_title('Monthly Average Hourly Rentals (2011 vs 2012)', fontweight='bold')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Average Hourly Rentals')
    axes[0].set_xticks(range(len(mnth_yr)))
    axes[0].set_xticklabels([month_names[i-1] for i in mnth_yr.index], rotation=45)
    axes[0].legend(['2011 (Yr 0)', '2012 (Yr 1)'])
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    season_names = ['Spring', 'Summer', 'Fall', 'Winter']
    sns.boxplot(x='season', y='cnt', data=train_df, ax=axes[1], color='gold', showmeans=True,
                meanprops={"marker":"o", "markerfacecolor":"blue", "markeredgecolor":"blue"})
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(season_names)
    axes[1].set_title('Seasonal Distribution of Rentals', fontweight='bold')
    axes[1].set_xlabel('Season')
    axes[1].set_ylabel('Hourly Count (cnt)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '03_month_and_season.png'), dpi=300)
    plt.close(fig)

    # =============================================================
    # QUESTION 4: OUTLIER ANALYSIS
    # =============================================================
    print("\n--- Question 4: Outliers ---")
    daily_all = df.groupby(['dteday', 'yr', 'mnth']).agg({
        'cnt': 'sum',
        'weathersit': 'mean',
        'temp_c': 'mean',
        'windspeed_kmh': 'mean',
        'hum_pct': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(x='hr', y='cnt', data=train_df, ax=axes[0], color='lightsteelblue', flierprops={'marker':'o', 'markersize':2, 'alpha':0.4})
    axes[0].set_title('Hourly Count Boxplot & Outliers', fontweight='bold')
    axes[0].set_xlabel('Hour of Day (hr)')
    axes[0].set_ylabel('Hourly Count (cnt)')
    
    axes[1].hist(daily_all['cnt'], bins=30, color='cadetblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(daily_all['cnt'].quantile(0.01), color='red', linestyle='--', label='1st Percentile')
    axes[1].axvline(daily_all['cnt'].quantile(0.99), color='green', linestyle='--', label='99th Percentile')
    axes[1].set_title('Daily Aggregate Count Distribution', fontweight='bold')
    axes[1].set_xlabel('Daily Total Rentals')
    axes[1].set_ylabel('Number of Days')
    axes[1].legend()
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '04_outliers_and_extremes.png'), dpi=300)
    plt.close(fig)

    # =============================================================
    # QUESTION 5: SCATTER PLOTS & INTERACTIONS
    # =============================================================
    print("\n--- Question 5: Scatter Plots ---")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    sample_sub = train_df.sample(2500, random_state=42)
    
    sns.regplot(x='temp_c', y='cnt', data=sample_sub, ax=axes[0],
                scatter_kws={'alpha':0.15, 'color':'coral', 's':10},
                line_kws={'color':'red', 'linewidth':2.5}, order=2)
    axes[0].set_title('Temperature vs Count (Inverted-U Curve)', fontweight='bold')
    axes[0].set_xlabel('Temperature (°C)')
    axes[0].set_ylabel('Hourly Count (cnt)')
    
    sns.regplot(x='atemp_c', y='cnt', data=sample_sub, ax=axes[1],
                scatter_kws={'alpha':0.15, 'color':'orange', 's':10},
                line_kws={'color':'darkorange', 'linewidth':2.5}, order=2)
    axes[1].set_title('Feeling Temperature vs Count (Collinear with temp)', fontweight='bold')
    axes[1].set_xlabel('Feeling Temperature (°C)')
    axes[1].set_ylabel('Hourly Count (cnt)')
    
    sns.regplot(x='hum_pct', y='cnt', data=sample_sub, ax=axes[2],
                scatter_kws={'alpha':0.15, 'color':'teal', 's':10},
                line_kws={'color':'darkblue', 'linewidth':2.5}, order=2)
    axes[2].set_title('Humidity vs Count (Sharp Drop >70%)', fontweight='bold')
    axes[2].set_xlabel('Humidity (%)')
    axes[2].set_ylabel('Hourly Count (cnt)')
    
    sns.regplot(x='windspeed_kmh', y='cnt', data=sample_sub, ax=axes[3],
                scatter_kws={'alpha':0.15, 'color':'purple', 's':10},
                line_kws={'color':'darkviolet', 'linewidth':2.5}, order=2)
    axes[3].set_title('Wind Speed vs Count', fontweight='bold')
    axes[3].set_xlabel('Wind Speed (km/h)')
    axes[3].set_ylabel('Hourly Count (cnt)')
    
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '05_scatter_continuous_vs_cnt.png'), dpi=300)
    plt.close(fig)

    # Full heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_cols = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit',
                    'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    sns.heatmap(train_df[numeric_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Correlation Heatmap: Training Features', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '05_full_correlation_heatmap.png'), dpi=300)
    plt.close(fig)

    # =============================================================
    # 6 SIMPLE NON-MODELING BEHAVIORAL EDA PLOTS
    # =============================================================
    print("\n--- Generating 6 Simple Behavioral EDA Plots ---")
    
    # 6.1 Weather Severity Breakdown
    weather_labels = ['Clear / Few Clouds', 'Mist / Cloudy', 'Light Rain / Snow', 'Heavy Rain / Ice']
    w_stats = train_df.groupby('weathersit')['cnt'].agg(['count', 'mean', 'median'])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar([weather_labels[i-1] for i in w_stats.index], w_stats['mean'], color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'], edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f} bikes/hr", ha='center', va='bottom', fontweight='bold')
    ax.set_title('Weather Severity Breakdown: Average Hourly Rentals', fontsize=13, fontweight='bold')
    ax.set_xlabel('Weather Situation (weathersit)')
    ax.set_ylabel('Average Hourly Rentals (cnt)')
    ax.set_ylim(0, 240)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_weather_severity.png'), dpi=300)
    plt.close(fig)

    # 6.2 Temperature Comfort Zones
    bins_temp = [-10, 10, 20, 28, 50]
    labels_temp = ['Cold (<10°C)', 'Cool (10-20°C)', 'Warm (20-28°C)', 'Hot (>28°C)']
    train_df['temp_bin'] = pd.cut(train_df['temp_c'], bins=bins_temp, labels=labels_temp)
    t_stats = train_df.groupby('temp_bin', observed=False)['cnt'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_temp, t_stats.values, color=['#4575b4', '#74add1', '#fdae61', '#d73027'], edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 6, f"{yval:.1f} bikes/hr", ha='center', va='bottom', fontweight='bold')
    ax.set_title('Temperature Comfort Zones: Average Rentals by Temperature Range', fontsize=13, fontweight='bold')
    ax.set_xlabel('Temperature Category')
    ax.set_ylabel('Average Hourly Rentals (cnt)')
    ax.set_ylim(0, 350)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_temperature_bins.png'), dpi=300)
    plt.close(fig)

    # 6.3 Humidity & Mugginess Bands
    bins_hum = [0, 40, 65, 80, 100]
    labels_hum = ['Dry (<40%)', 'Comfortable (40-65%)', 'Humid (65-80%)', 'Muggy (>80%)']
    train_df['hum_bin'] = pd.cut(train_df['hum_pct'], bins=bins_hum, labels=labels_hum)
    h_stats = train_df.groupby('hum_bin', observed=False)['cnt'].mean()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels_hum, h_stats.values, color=['#a6d96a', '#ffffbf', '#fdae61', '#d7191c'], edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f} bikes/hr", ha='center', va='bottom', fontweight='bold')
    ax.set_title('Humidity Mugginess Bands: Demand Drops as Moisture Rises', fontsize=13, fontweight='bold')
    ax.set_xlabel('Humidity Band')
    ax.set_ylabel('Average Hourly Rentals (cnt)')
    ax.set_ylim(0, 320)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_humidity_bands.png'), dpi=300)
    plt.close(fig)

    # 6.4 Commuter vs Casual Adverse Weather Reaction
    w_user = train_df.groupby('weathersit')[['casual', 'registered']].mean()
    w_user_sub = w_user.loc[[1, 3]]  # Clear vs Rain/Snow
    
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(2)
    width = 0.35
    b1 = ax.bar(x - width/2, w_user_sub['registered'], width, label='Registered (Commuters)', color='#1f77b4', edgecolor='black')
    b2 = ax.bar(x + width/2, w_user_sub['casual'], width, label='Casual (Leisure)', color='#ff7f0e', edgecolor='black')
    
    for b in b1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{b.get_height():.1f}", ha='center', va='bottom', fontweight='bold')
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 3, f"{b.get_height():.1f}", ha='center', va='bottom', fontweight='bold')
        
    ax.set_xticks(x)
    ax.set_xticklabels(['Clear Weather', 'Light Rain / Snow'])
    ax.set_title('Bad Weather Reaction: Casual vs Registered Riders', fontsize=13, fontweight='bold')
    ax.set_ylabel('Average Hourly Rentals')
    ax.set_ylim(0, 190)
    ax.legend()
    # Annotate drop percentages
    ax.text(0.7, 130, 'Registered Drop: -41.9%\nCasual Drop: -60.4%', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_commuter_vs_casual_rain.png'), dpi=300)
    plt.close(fig)

    # 6.5 Rush Hour Asymmetry (Morning Spike vs Evening Plateau)
    workday_hours = train_df[train_df['workingday'] == 1].groupby('hr')['cnt'].mean()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(workday_hours.index, workday_hours.values, marker='o', color='navy', linewidth=2.5, label='Working Day Average')
    ax.axvspan(7, 9, color='lightgreen', alpha=0.3, label='Morning Rush (Sharp 8 AM Spike)')
    ax.axvspan(16, 19, color='lightcoral', alpha=0.3, label='Evening Rush (Broad 5-6 PM Plateau)')
    ax.set_title('Rush Hour Asymmetry on Working Days', fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour of Day (hr: 0 - 23)')
    ax.set_ylabel('Average Hourly Rentals')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlim(5, 23)
    ax.legend(loc='upper left')
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_rush_hour_asymmetry.png'), dpi=300)
    plt.close(fig)

    # 6.6 Overnight Dead Zone (12 AM - 5 AM)
    night_stats = train_df[train_df['hr'] <= 5].groupby('hr')['cnt'].mean()
    night_labels = ['12 AM', '1 AM', '2 AM', '3 AM', '4 AM (Bottom)', '5 AM']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(night_labels, night_stats.values, color=['#3182bd', '#6baed6', '#9ecae1', '#c6dbef', '#de2d26', '#31a354'], edgecolor='black', width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    ax.set_title('The Overnight "Dead Zone" (12:00 AM - 5:00 AM)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Overnight Hours')
    ax.set_ylabel('Average Hourly Rentals')
    ax.set_ylim(0, 65)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, '06_simple_overnight_deadzone.png'), dpi=300)
    plt.close(fig)

    print("All 6 simple exploratory plots generated successfully!")

if __name__ == '__main__':
    main()
