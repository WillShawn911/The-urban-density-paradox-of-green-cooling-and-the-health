import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm

# ==========================================
# 配置路径
# ==========================================
# 输入：阶段三清洗后的数据
INPUT_DIR = "/Users/stonerose/Downloads/100cities/results_2sfca_cleaned"
# 输出：阶段四公平性评估结果
OUTPUT_DIR = "/Users/stonerose/Downloads/100cities/phase4_equity_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def calc_population_weighted_gini(df, score_col='accessibility_score', pop_col='population'):
    """
    计算人口加权的基尼系数 (Population-weighted Gini Coefficient)
    """
    # 过滤掉没有人口的网格 (避免除以 0)
    df_valid = df[(df[pop_col] > 0) & (df[score_col] >= 0)].copy()
    
    if df_valid.empty or df_valid[score_col].sum() == 0:
        return np.nan # 如果全城得分为0，基尼系数无意义
        
    # 1. 按照可达性得分从小到大排序 (从“最穷”到“最富”)
    df_sorted = df_valid.sort_values(by=score_col).reset_index(drop=True)
    
    # 2. 计算每个网格拥有的“总绿地资产” = 人均得分 * 人口
    df_sorted['total_green_assets'] = df_sorted[score_col] * df_sorted[pop_col]
    
    # 3. 计算人口和绿地资产的累计百分比 (累积比例 X 和 Y)
    cum_pop = df_sorted[pop_col].cumsum()
    cum_assets = df_sorted['total_green_assets'].cumsum()
    
    X = cum_pop / cum_pop.iloc[-1]
    Y = cum_assets / cum_assets.iloc[-1]
    
    # 4. 在最前面插入 0，闭合洛伦兹曲线起点 (0,0)
    X = np.insert(X.values, 0, 0)
    Y = np.insert(Y.values, 0, 0)
    
    # 5. 使用梯形面积法计算基尼系数：Gini = 1 - sum( (X_i - X_{i-1}) * (Y_i + Y_{i-1}) )
    B_area = np.sum((X[1:] - X[:-1]) * (Y[1:] + Y[:-1]))
    gini = 1 - B_area
    
    return gini

def main():
    print("="*70)
    print(" ⚖️ 第四阶段：全球百城绿地空间公平性 (Gini) 计算启动")
    print("="*70)

    input_files = glob.glob(os.path.join(INPUT_DIR, "*_accessibility_cleaned.geojson"))
    if not input_files:
        print(f"❌ 找不到清洗后的结果文件: {INPUT_DIR}")
        return

    print(f"---> 准备计算 {len(input_files)} 个城市的基尼系数...\n")

    equity_results = []

    for filepath in tqdm(input_files, desc="正在计算基尼系数"):
        city_name = os.path.basename(filepath).replace("_accessibility_cleaned.geojson", "")
        
        try:
            # 只加载需要的列以节省内存并加速
            gdf = gpd.read_file(filepath, ignore_geometry=True)
            
            total_pop = gdf['population'].sum()
            mean_score = gdf['accessibility_score'].mean()
            median_score = gdf['accessibility_score'].median()
            
            # 计算全城 0 得分人口的比例 (完全没有绿地的人)
            pop_zero_access = gdf[gdf['accessibility_score'] == 0]['population'].sum()
            zero_access_pct = (pop_zero_access / total_pop) * 100 if total_pop > 0 else 0
            
            # 计算基尼系数
            gini_index = calc_population_weighted_gini(gdf, 'accessibility_score', 'population')
            
            equity_results.append({
                'City': city_name,
                'Total_Population': int(total_pop),
                'Mean_Accessibility': round(mean_score, 2),
                'Median_Accessibility': round(median_score, 2),
                'Zero_Access_Pop_Pct': round(zero_access_pct, 2),
                'Gini_Coefficient': round(gini_index, 4) if pd.notna(gini_index) else None
            })

        except Exception as e:
            print(f"\n❌ [{city_name}] 计算失败: {e}")

    # 生成最终的数据表
    df_equity = pd.DataFrame(equity_results)
    
    # 按基尼系数从高到低排序 (不公平程度从大到小)
    df_equity = df_equity.sort_values(by='Gini_Coefficient', ascending=False)
    
    # 保存为 CSV
    output_csv = os.path.join(OUTPUT_DIR, "Global_118_Cities_Equity_Metrics.csv")
    df_equity.to_csv(output_csv, index=False)
    
    print("\n" + "="*70)
    print(f"🎉 全球 {len(df_equity)} 城公平性指标计算完成！")
    print(f"全球平均基尼系数: {df_equity['Gini_Coefficient'].mean():.4f}")
    print(f"📄 核心汇总表已保存至: \n{output_csv}")
    print("="*70)
    print("💡 下一步建议：打开这份 CSV，看看哪些城市最公平，哪些最不公平！")

if __name__ == "__main__":
    main()