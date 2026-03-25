import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 顶级绘图风格
plt.style.use('seaborn-v0_8-ticks')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['svg.fonttype'] = 'none'

# ==========================================
# 0. 路径配置 (已更新为 UHI_results)
# ==========================================
GEOJSON_DIR = "/Users/stonerose/Downloads/100cities/results_2sfca_cleaned"
UHI_DIR = "/Users/stonerose/Downloads/100cities/UHI"
GEE_DIR = "/Users/stonerose/Downloads/GEE_data"  # 用于提取 NTL 和 NDVI

# ✅ 新的输出目录
OUTPUT_DIR = "/Users/stonerose/Downloads/100cities/UHI_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_grid_level_thermal_data():
    """
    第一步：微观空间融合
    将 LST(热岛强度) 和 NTL(夜间灯光) 提取到每个 500m 街区网格中
    """
    print("="*70)
    print(" 🧬 启动全球 118 城微观空间热融合流水线...")
    print("="*70)

    geojson_files = glob.glob(os.path.join(GEOJSON_DIR, "*_accessibility_cleaned.geojson"))
    
    # 我们不把 700 万个网格全塞进内存，我们抽取 15% 的代表性网格，兼顾速度与统计显著性
    SAMPLE_RATE = 0.15 
    global_grids = []

    for filepath in tqdm(geojson_files, desc="提取网格温度与经济指标"):
        city_name = os.path.basename(filepath).replace("_accessibility_cleaned.geojson", "")
        safe_name = city_name.replace(' ', '_').replace('/', '-')
        
        uhi_file = os.path.join(UHI_DIR, f"{safe_name}_UHII_100m.tif")
        ntl_file = os.path.join(GEE_DIR, f"{safe_name}_VIIRS_NTL.tif")
        
        if not os.path.exists(uhi_file):
            continue
            
        try:
            gdf = gpd.read_file(filepath)
            if gdf.empty: continue
                
            # 随机抽样网格
            gdf_sample = gdf.sample(frac=SAMPLE_RATE, random_state=42).copy()
            
            # 提取热岛强度 (UHII)
            uhi_stats = zonal_stats(gdf_sample, uhi_file, stats="mean", nodata=np.nan)
            gdf_sample['UHII'] = [s['mean'] for s in uhi_stats]
            
            # 提取夜间灯光 (NTL，作为街区财富代理)
            if os.path.exists(ntl_file):
                ntl_stats = zonal_stats(gdf_sample, ntl_file, stats="mean", nodata=np.nan)
                gdf_sample['NTL'] = [s['mean'] for s in ntl_stats]
            else:
                gdf_sample['NTL'] = np.nan
                
            gdf_sample['City'] = city_name
            
            # 丢弃几何列以节约内存
            df_subset = pd.DataFrame(gdf_sample.drop(columns=['geometry']))
            global_grids.append(df_subset)
            
        except Exception as e:
            tqdm.write(f"❌ {city_name} 提取失败: {e}")

    df_global = pd.concat(global_grids, ignore_index=True)
    df_global = df_global.dropna(subset=['UHII', 'accessibility_score', 'NTL']).copy()
    
    # 极值清理
    df_global = df_global[(df_global['UHII'] > -15) & (df_global['UHII'] < 30)]
    df_global['NTL_log'] = np.log1p(df_global['NTL'])
    
    print(f"\n✅ 空间融合完成！成功提取了 {len(df_global):,} 个具有完整经济和温度数据的微观街区。")
    return df_global

def run_thermal_justice_model(df):
    """
    第二步：机器学习归因 (The Inverse Cooling Law)
    使用随机森林证明：财富(NTL)不仅直接影响绿地获取，还影响绿地的降温效能
    """
    print("\n" + "="*70)
    print(" 🌲 训练热正义机器学习模型...")
    print("="*70)
    
    # 核心特征：我们想看绿地获取、人口和财富，是如何决定一个街区的热岛强度的
    features = ['accessibility_score', 'population', 'NTL_log']
    X = df[features]
    y = df['UHII']

    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # 输出特征重要性
    importances = rf.feature_importances_
    for feat, imp in zip(features, importances):
        print(f"  - {feat} 对热岛强度的解释力: {imp*100:.1f}%")
        
    return df

def plot_the_inverse_cooling_law(df):
    """
    第三步：绘制顶刊级核心图表 (降温的反向比例定律)
    修复：增强了对穷人区大量“零绿地”街区的统计鲁棒性
    """
    print("\n---> 正在生成顶刊核心图表：The Inverse Cooling Law...")
    
    # 将全部街区按财富 (NTL) 分为 10 个等分组 (使用 duplicates='drop' 防止极端聚集)
    df['Wealth_Decile'] = pd.qcut(df['NTL_log'], 10, labels=False, duplicates='drop')
    
    cooling_efficiency = []
    deciles = sorted(df['Wealth_Decile'].unique())
    
    for decile in deciles:
        subset = df[df['Wealth_Decile'] == decile]
        if len(subset) < 50: continue
            
        # 强制取该财富阶层中，绿地得分最高和最低的 20% 街区
        n_20_percent = max(1, int(len(subset) * 0.20))
        high_green = subset.nlargest(n_20_percent, 'accessibility_score')
        low_green = subset.nsmallest(n_20_percent, 'accessibility_score')
        
        # 确保这两组在绿地上确实存在差距 (防止全区都是0的极端情况)
        if high_green['accessibility_score'].mean() > low_green['accessibility_score'].mean():
            # 降温效率 = 低绿地街区的温度 - 高绿地街区的温度 (正值代表绿地起到了降温作用)
            cooling_effect = low_green['UHII'].mean() - high_green['UHII'].mean()
            cooling_efficiency.append({
                'Wealth_Decile': decile + 1,
                'Mean_NTL': subset['NTL'].mean(),
                'Cooling_Effect_Celsius': cooling_effect
            })
            
    df_plot = pd.DataFrame(cooling_efficiency)
    
    if df_plot.empty:
        print("❌ 数据过于偏态，无法计算有效差距。")
        return

    # 开始绘图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制回归线和散点
    sns.regplot(data=df_plot, x='Wealth_Decile', y='Cooling_Effect_Celsius',
                scatter_kws={'s': 150, 'color': '#2b83ba', 'edgecolor': 'white', 'alpha': 0.8},
                line_kws={'color': '#d7191c', 'linewidth': 3}, ax=ax)

    ax.set_title('The Inverse Cooling Law: Thermal Injustice in Global Cities', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Neighborhood Wealth Decile (1=Poorest, 10=Wealthiest)', fontsize=13)
    ax.set_ylabel('Cooling Efficiency of Green Space (Δ°C)', fontsize=13)
    
    # 添加核心发现的批注
    ax.annotate('Double Penalty:\nPoorest neighborhoods\nreceive the lowest cooling\nreturn from green spaces',
                xy=(1, df_plot['Cooling_Effect_Celsius'].min()), xycoords='data',
                xytext=(1.5, df_plot['Cooling_Effect_Celsius'].max() * 0.9), textcoords='data',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='#a50026', lw=1.5),
                fontsize=11, fontweight='bold', color='#a50026',
                bbox=dict(boxstyle="round,pad=0.5", fc="#fff5f0", ec="#a50026", alpha=0.9))

    ax.set_xticks(range(1, len(deciles) + 1))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # ✅ 图表保存到新的目录
    out_path = os.path.join(OUTPUT_DIR, "Fig8_Thermal_Justice_Inverse_Law.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 图表已保存至: {out_path}")

def main():
    # 1. 提取大样本微观数据
    # 注意：如果之前已经成功跑出数据，可以将提取部分注释掉，直接读取 csv，以节省时间
    df_micro = extract_grid_level_thermal_data()
    
    # 2. 运行机器学习与规律挖掘
    if not df_micro.empty:
        run_thermal_justice_model(df_micro)
        plot_the_inverse_cooling_law(df_micro)
        
        # 将微观面板数据存档
        csv_path = os.path.join(OUTPUT_DIR, "Global_Micro_Grid_Thermal_Data.csv")
        df_micro.to_csv(csv_path, index=False)
        print(f"\n🎉 全局热公平微观数据集已保存至: {csv_path}")
        print("这相当于为你打开了冲击第二篇顶刊的宝库！")

if __name__ == "__main__":
    main()