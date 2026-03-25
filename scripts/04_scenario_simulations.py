import os
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import matplotlib.patheffects as pe

# ==========================================
# 路径配置
# ==========================================
CLEANED_DIR = "/Users/stonerose/Downloads/100cities/results_2sfca_cleaned"
FIGURES_DIR = "/Users/stonerose/Downloads/100cities/phase5_figures"
OUTPUT_DIR = "/Users/stonerose/Downloads/100cities/phase5_simulation_results"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 顶级绘图风格
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# 干预预算：新增全市当前绿地总容量的 10%
INCREASE_RATIO = 0.10  

def calc_population_weighted_gini(df, score_col='accessibility_score', pop_col='population'):
    """计算人口加权基尼系数 (核心数学引擎)"""
    df_valid = df[(df[pop_col] > 0) & (df[score_col] >= 0)].copy()
    if df_valid.empty or df_valid[score_col].sum() == 0: return np.nan
        
    df_sorted = df_valid.sort_values(by=score_col).reset_index(drop=True)
    df_sorted['total_assets'] = df_sorted[score_col] * df_sorted[pop_col]
    
    cum_pop = df_sorted[pop_col].cumsum()
    cum_assets = df_sorted['total_assets'].cumsum()
    
    X = cum_pop / cum_pop.iloc[-1]
    Y = cum_assets / cum_assets.iloc[-1]
    X = np.insert(X.values, 0, 0)
    Y = np.insert(Y.values, 0, 0)
    
    B_area = np.sum((X[1:] - X[:-1]) * (Y[1:] + Y[:-1]))
    return 1 - B_area

def run_simulation(city_name):
    """对目标城市运行 3 种规划情景模拟"""
    filepath = os.path.join(CLEANED_DIR, f"{city_name}_accessibility_cleaned.geojson")
    if not os.path.exists(filepath): return None
        
    gdf = gpd.read_file(filepath, ignore_geometry=True)
    
    # 基础状态 (Baseline)
    baseline_gini = calc_population_weighted_gini(gdf, 'accessibility_score')
    
    # 计算当前全市拥有的“绿地总资产” (用于计算新增预算)
    gdf['current_assets'] = gdf['accessibility_score'] * gdf['population']
    total_assets = gdf['current_assets'].sum()
    budget_assets = total_assets * INCREASE_RATIO
    
    if budget_assets <= 0: return None

    # ------------------------------------------------------------------
    # 情景 1: 锦上添花 (Mega-Park at edges / Pro-Rich)
    # ------------------------------------------------------------------
    gdf_s1 = gdf.copy()
    threshold = gdf_s1['accessibility_score'].quantile(0.80)
    rich_grids = gdf_s1[gdf_s1['accessibility_score'] >= threshold]
    rich_pop_sum = rich_grids['population'].sum()
    if rich_pop_sum > 0:
        added_score = (budget_assets * (rich_grids['population'] / rich_pop_sum)) / rich_grids['population']
        gdf_s1.loc[rich_grids.index, 'accessibility_score'] += added_score
    s1_gini = calc_population_weighted_gini(gdf_s1, 'accessibility_score')

    # ------------------------------------------------------------------
    # 情景 2: 均等发展 (Equal Distribution)
    # ------------------------------------------------------------------
    gdf_s2 = gdf.copy()
    total_pop = gdf_s2['population'].sum()
    added_score_per_capita = budget_assets / total_pop
    gdf_s2['accessibility_score'] += added_score_per_capita
    s2_gini = calc_population_weighted_gini(gdf_s2, 'accessibility_score')

    # ------------------------------------------------------------------
    # 情景 3: 靶向口袋公园 (Pro-Poor Pocket Parks)
    # ------------------------------------------------------------------
    gdf_s3 = gdf.copy()
    poor_threshold = gdf_s3['accessibility_score'].quantile(0.50)
    poor_grids = gdf_s3[gdf_s3['accessibility_score'] <= poor_threshold]
    poor_pop_sum = poor_grids['population'].sum()
    if poor_pop_sum > 0:
        added_score_poor = (budget_assets * (poor_grids['population'] / poor_pop_sum)) / poor_grids['population']
        gdf_s3.loc[poor_grids.index, 'accessibility_score'] += added_score_poor
    s3_gini = calc_population_weighted_gini(gdf_s3, 'accessibility_score')

    return {
        'City': city_name.replace('_', ' '),
        'Baseline': baseline_gini,
        'Mega-Park (Pro-Rich)': s1_gini,
        'Equal Greening': s2_gini,
        'Pocket Parks (Pro-Poor)': s3_gini
    }

def main():
    print("="*70)
    print(" 🌍 启动全球 118 城绿地干预全样本情景模拟 (Nature 流星雨终极版)")
    print("="*70)
    
    # 获取所有城市
    city_files = [f for f in os.listdir(CLEANED_DIR) if f.endswith("_accessibility_cleaned.geojson")]
    all_results = []
    
    for f in tqdm(city_files, desc="运行情景模拟器"):
        city = f.replace("_accessibility_cleaned.geojson", "")
        res = run_simulation(city)
        if res:
            all_results.append(res)
            
    df_all = pd.DataFrame(all_results)
    
    # 导出全样本数据表 (作为补充材料)
    csv_out = os.path.join(OUTPUT_DIR, "All_118_Cities_Simulation_Results.csv")
    df_all.to_csv(csv_out, index=False)
    print(f"\n✅ 118 城全样本模拟完成！详细数据已保存至附录表: {csv_out}")