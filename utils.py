import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import t

def get_significance_level(f_value, f_table_5, f_table_1):
    """
    Menentukan tingkat signifikansi berdasarkan perbandingan F-hitung dengan F-tabel
    """
    if f_value > f_table_1:
        return 0.01, "Berbeda Sangat Nyata (P < 0.01)"
    elif f_value > f_table_5:
        return 0.05, "Berbeda Nyata (0.01 < P < 0.05)"
    else:
        return 1.0, "Tidak Ada Perbedaan Nyata (P > 0.05)"

def calculate_bnt(data, ms_error, alpha=0.05):
    """
    Menghitung Beda Nyata Terkecil (BNT/LSD)
    """
    try:
        # Convert numeric values
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data['treatment'] = data['treatment'].astype(str)
        ms_error = float(ms_error)
        alpha = float(alpha)
        
        # Drop any NaN values
        data = data.dropna(subset=['value'])
        
        if len(data) == 0:
            return {'error': 'No valid numeric data found'}
            
        df_error = len(data) - len(data['treatment'].unique())
        t_value = float(t.ppf(1 - alpha/2, df_error))
        treatments = data.groupby('treatment')
        n = float(treatments.size().min())
        
        bnt_value = float(t_value * np.sqrt((2 * ms_error) / n))
        means = treatments['value'].mean().sort_values(ascending=False)
        
        comparison_matrix = pd.DataFrame(index=means.index, columns=means.index)
        notations = {str(treatment): '' for treatment in means.index}
        
        for i, treatment1 in enumerate(means.index):
            for treatment2 in means.index[i+1:]:
                diff = abs(float(means[treatment1]) - float(means[treatment2]))
                significant = diff > bnt_value
                comparison_matrix.loc[treatment1, treatment2] = 'S' if significant else 'NS'
                comparison_matrix.loc[treatment2, treatment1] = 'S' if significant else 'NS'
        
        current_notation = 'a'
        for treatment in means.index:
            if not notations[str(treatment)]:
                notations[str(treatment)] = current_notation
                for other_treatment in means.index:
                    if other_treatment != treatment and comparison_matrix.loc[treatment, other_treatment] == 'NS':
                        notations[str(other_treatment)] = current_notation
                current_notation = chr(ord(current_notation) + 1)
        
        return {
            'bnt_value': float(bnt_value),
            'means': {str(k): float(v) for k, v in means.round(3).to_dict().items()},
            'notations': notations,
            'comparison_matrix': comparison_matrix.to_dict()
        }
    except Exception as e:
        return {'error': str(e)}

def calculate_duncan(data, ms_error, alpha=0.05):
    """
    Menghitung Uji Duncan
    """
    try:
        # Convert numeric values
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data['treatment'] = data['treatment'].astype(str)
        ms_error = float(ms_error)
        alpha = float(alpha)
        
        # Drop any NaN values
        data = data.dropna(subset=['value'])
        
        if len(data) == 0:
            return {'error': 'No valid numeric data found'}
            
        df_error = len(data) - len(data['treatment'].unique())
        treatments = data.groupby('treatment')
        n = float(treatments.size().min())
        
        # Calculate means and sort
        means = treatments['value'].mean().sort_values(ascending=False)
        
        # Get p values for Duncan's test
        p_values = range(2, len(means) + 1)
        
        # Calculate Duncan values
        duncan_values = []
        for p in p_values:
            rp = float(stats.studentized_range.ppf(q=1-alpha, rng=p, df=df_error))
            duncan = float(rp * np.sqrt(ms_error / n))
            duncan_values.append(duncan)
        
        # Initialize comparison matrix and notations
        comparison_matrix = pd.DataFrame(index=means.index, columns=means.index)
        notations = {str(treatment): '' for treatment in means.index}
        
        # Perform comparisons
        for i, treatment1 in enumerate(means.index):
            for j, treatment2 in enumerate(means.index[i+1:], i+1):
                diff = abs(float(means[treatment1]) - float(means[treatment2]))
                p = j - i + 1
                significant = diff > duncan_values[p-2]
                comparison_matrix.loc[treatment1, treatment2] = 'S' if significant else 'NS'
                comparison_matrix.loc[treatment2, treatment1] = 'S' if significant else 'NS'
        
        # Assign notations
        current_notation = 'a'
        for treatment in means.index:
            if not notations[str(treatment)]:
                notations[str(treatment)] = current_notation
                for other_treatment in means.index:
                    if other_treatment != treatment and comparison_matrix.loc[treatment, other_treatment] == 'NS':
                        notations[str(other_treatment)] = current_notation
                current_notation = chr(ord(current_notation) + 1)
        
        return {
            'duncan_values': [float(x) for x in duncan_values],
            'means': {str(k): float(v) for k, v in means.round(3).to_dict().items()},
            'notations': notations,
            'comparison_matrix': comparison_matrix.to_dict()
        }
    except Exception as e:
        return {'error': str(e)}

def process_ral(data, post_hoc=None):
    """
    Analisis RAL (Rancangan Acak Lengkap)
    """
    try:
        df = pd.DataFrame(data)
        # Ensure treatment is string and value is float
        df['treatment'] = df['treatment'].astype(str)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop any NaN values
        df = df.dropna(subset=['value'])
        
        if len(df) == 0:
            return {'error': 'No valid numeric data found'}
        
        # Statistik deskriptif
        summary = df.groupby('treatment')['value'].agg(['mean', 'std', 'count']).round(3)
        
        # ANOVA calculations
        treatments = df.groupby('treatment')
        grand_mean = float(df['value'].mean())
        
        # Derajat bebas
        df_treatment = len(treatments) - 1
        df_error = len(df) - len(treatments)
        df_total = len(df) - 1
        
        if df_error <= 0:
            return {'error': 'Not enough data for analysis'}
        
        # Jumlah Kuadrat
        ss_total = float(sum((df['value'] - grand_mean) ** 2))
        ss_treatment = float(sum(len(group) * (float(group['value'].mean()) - grand_mean) ** 2 
                         for _, group in treatments))
        ss_error = float(ss_total - ss_treatment)
        
        # Kuadrat Tengah
        ms_treatment = float(ss_treatment / df_treatment)
        ms_error = float(ss_error / df_error)
        
        # F-value dan P-value
        f_value = float(ms_treatment / ms_error)
        try:
            p_value = float(1 - stats.f.cdf(f_value, df_treatment, df_error))
        except:
            p_value = 1.0  # Default to no significance if calculation fails
        
        # Determine significance level and message
        f_table_5 = stats.f.ppf(0.95, df_treatment, df_error)
        f_table_1 = stats.f.ppf(0.99, df_treatment, df_error)
        alpha, significance_msg = get_significance_level(f_value, f_table_5, f_table_1)
        
        # Koefisien Keragaman
        cv = float((np.sqrt(ms_error) / grand_mean) * 100)
        
        anova_table = {
            'Source': ['Treatment', 'Error', 'Total'],
            'DF': [int(df_treatment), int(df_error), int(df_total)],
            'SS': [float(round(ss_treatment, 3)), float(round(ss_error, 3)), float(round(ss_total, 3))],
            'MS': [float(round(ms_treatment, 3)), float(round(ms_error, 3)), '-'],
            'F': [float(round(f_value, 3)), '-', '-'],
            'Ftab5': [float(round(f_table_5, 3)), '-', '-'],
            'Ftab1': [float(round(f_table_1, 3)), '-', '-'],
            'P': [float(round(p_value, 4)), '-', '-']
        }
        
        result = {
            'summary': {str(k): {'mean': float(v['mean']), 'std': float(v['std']), 'count': int(v['count'])} 
                       for k, v in summary.to_dict('index').items()},
            'anova': anova_table,
            'cv': float(round(cv, 2)),
            'significance': significance_msg,
            'p_value': float(p_value)
        }
        
        # Post-hoc test if treatments are significantly different and post_hoc is specified
        if float(p_value) <= float(alpha) and post_hoc:  # Only do post-hoc if requested
            if post_hoc == 'bnt':
                result['post_hoc'] = calculate_bnt(df, ms_error, alpha)
                result['post_hoc_type'] = 'BNT'
            else:
                result['post_hoc'] = calculate_duncan(df, ms_error, alpha)
                result['post_hoc_type'] = 'Duncan'
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {'error': f"{error_msg}\n{stack_trace}"}

def process_rak(data, post_hoc=None):
    """
    Analisis RAK (Rancangan Acak Kelompok)
    """
    try:
        df = pd.DataFrame(data)
        # Convert replication to block
        df['block'] = df['replication']
        # Ensure treatment and block are strings, value is float
        df['treatment'] = df['treatment'].astype(str)
        df['block'] = df['block'].astype(str)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Drop any NaN values
        df = df.dropna(subset=['value'])
        
        if len(df) == 0:
            return {'error': 'No valid numeric data found'}
        
        # Statistik deskriptif
        summary = df.groupby('treatment')['value'].agg(['mean', 'std', 'count']).round(3)
        
        # ANOVA calculations
        treatments = df['treatment'].unique()
        blocks = df['block'].unique()
        grand_mean = float(df['value'].mean())
        
        # Derajat bebas
        df_treatment = len(treatments) - 1
        df_block = len(blocks) - 1
        df_error = (len(treatments) - 1) * (len(blocks) - 1)
        df_total = len(df) - 1
        
        if df_error <= 0:
            return {'error': 'Not enough data for analysis'}
        
        # Jumlah Kuadrat
        ss_total = float(sum((df['value'] - grand_mean) ** 2))
        
        # Treatment SS
        treatment_means = df.groupby('treatment')['value'].mean()
        ss_treatment = float(sum(len(blocks) * (treatment_means - grand_mean) ** 2))
        
        # Block SS
        block_means = df.groupby('block')['value'].mean()
        ss_block = float(sum(len(treatments) * (block_means - grand_mean) ** 2))
        
        # Error SS
        ss_error = float(ss_total - ss_treatment - ss_block)
        
        # Kuadrat Tengah
        ms_treatment = float(ss_treatment / df_treatment)
        ms_block = float(ss_block / df_block)
        ms_error = float(ss_error / df_error)
        
        # F-values
        f_value_treatment = float(ms_treatment / ms_error)
        f_value_block = float(ms_block / ms_error)
        
        # P-values
        try:
            p_value_treatment = float(1 - stats.f.cdf(f_value_treatment, df_treatment, df_error))
            p_value_block = float(1 - stats.f.cdf(f_value_block, df_block, df_error))
        except:
            p_value_treatment = 1.0
            p_value_block = 1.0
        
        # Determine significance level and message
        f_table_5_treatment = stats.f.ppf(0.95, df_treatment, df_error)
        f_table_1_treatment = stats.f.ppf(0.99, df_treatment, df_error)
        f_table_5_block = stats.f.ppf(0.95, df_block, df_error)
        f_table_1_block = stats.f.ppf(0.99, df_block, df_error)
        alpha_treatment, significance_msg_treatment = get_significance_level(f_value_treatment, f_table_5_treatment, f_table_1_treatment)
        alpha_block, significance_msg_block = get_significance_level(f_value_block, f_table_5_block, f_table_1_block)
        
        # Koefisien Keragaman
        cv = float((np.sqrt(ms_error) / grand_mean) * 100)
        
        anova_table = {
            'Source': ['Treatment', 'Block', 'Error', 'Total'],
            'DF': [int(df_treatment), int(df_block), int(df_error), int(df_total)],
            'SS': [float(round(ss_treatment, 3)), float(round(ss_block, 3)), 
                  float(round(ss_error, 3)), float(round(ss_total, 3))],
            'MS': [float(round(ms_treatment, 3)), float(round(ms_block, 3)), 
                  float(round(ms_error, 3)), '-'],
            'F': [float(round(f_value_treatment, 3)), float(round(f_value_block, 3)), '-', '-'],
            'Ftab5': [float(round(f_table_5_treatment, 3)), float(round(f_table_5_block, 3)), '-', '-'],
            'Ftab1': [float(round(f_table_1_treatment, 3)), float(round(f_table_1_block, 3)), '-', '-'],
            'P': [float(round(p_value_treatment, 4)), float(round(p_value_block, 4)), '-', '-']
        }
        
        result = {
            'summary': {str(k): {'mean': float(v['mean']), 'std': float(v['std']), 'count': int(v['count'])} 
                       for k, v in summary.to_dict('index').items()},
            'anova': anova_table,
            'cv': float(round(cv, 2)),
            'significance': {
                'treatment': significance_msg_treatment,
                'block': significance_msg_block
            },
            'p_value': {
                'treatment': float(p_value_treatment),
                'block': float(p_value_block)
            }
        }
        
        # Post-hoc test if treatments are significantly different and post_hoc is specified
        if float(p_value_treatment) <= float(alpha_treatment) and post_hoc:  # Only do post-hoc if requested
            if post_hoc == 'bnt':
                result['post_hoc'] = calculate_bnt(df, ms_error, alpha_treatment)
                result['post_hoc_type'] = 'BNT'
            else:
                result['post_hoc'] = calculate_duncan(df, ms_error, alpha_treatment)
                result['post_hoc_type'] = 'Duncan'
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {'error': f"{error_msg}\n{stack_trace}"}

def process_ral_factorial(data, post_hoc='bnt'):
    """
    Analisis RAL Faktorial
    """
    try:
        df = pd.DataFrame(data)
        # Ensure all numeric values are properly converted
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['replication'] = pd.to_numeric(df['replication'], errors='coerce')
        
        if not all(col in df.columns for col in ['factor_a', 'factor_b', 'replication', 'value']):
            return {'error': 'Data tidak valid, kolom yang diperlukan: factor_a, factor_b, replication, value'}
        
        # Drop any rows with NaN values after conversion
        df = df.dropna()
        
        if len(df) == 0:
            return {'error': 'No valid data after cleaning'}
        
        # Statistik deskriptif
        summary = df.groupby(['factor_a', 'factor_b'])['value'].agg(['mean', 'std', 'count']).round(3)
        summary_dict = {}
        for idx, row in summary.iterrows():
            key = f"{idx[0]} x {idx[1]}"  # Menggunakan format "A x B"
            summary_dict[key] = {
                'mean': 0 if pd.isna(row['mean']) else float(row['mean']),
                'std': 0 if pd.isna(row['std']) else float(row['std']),
                'count': int(row['count'])
            }
        
        # ANOVA calculations
        factor_a_levels = df['factor_a'].unique()
        factor_b_levels = df['factor_b'].unique()
        replications = df['replication'].unique()
        
        # Derajat bebas
        df_factor_a = len(factor_a_levels) - 1
        df_factor_b = len(factor_b_levels) - 1
        df_interaction = df_factor_a * df_factor_b
        df_error = len(df) - (len(factor_a_levels) * len(factor_b_levels))
        df_total = len(df) - 1
        
        # Convert all values to float and calculate
        values = df['value'].astype(float)
        grand_mean = float(values.mean())
        
        # Jumlah Kuadrat
        ss_total = float(np.sum((values - grand_mean) ** 2))
        
        # SS Factor A
        ss_factor_a = float(sum(len(group) * (group['value'].mean() - grand_mean) ** 2 
                         for _, group in df.groupby('factor_a')))
        
        # SS Factor B
        ss_factor_b = float(sum(len(group) * (group['value'].mean() - grand_mean) ** 2 
                         for _, group in df.groupby('factor_b')))
        
        # SS Interaction
        ss_interaction = float(sum(len(group) * (group['value'].mean() - grand_mean) ** 2 
                           for _, group in df.groupby(['factor_a', 'factor_b']))) - ss_factor_a - ss_factor_b
        
        # SS Error
        ss_error = float(ss_total - ss_factor_a - ss_factor_b - ss_interaction)
        
        # Mean Squares
        ms_factor_a = ss_factor_a / df_factor_a if df_factor_a > 0 else 0
        ms_factor_b = ss_factor_b / df_factor_b if df_factor_b > 0 else 0
        ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
        ms_error = ss_error / df_error if df_error > 0 else 0
        
        # F-values
        f_factor_a = ms_factor_a / ms_error if ms_error > 0 else 0
        f_factor_b = ms_factor_b / ms_error if ms_error > 0 else 0
        f_interaction = ms_interaction / ms_error if ms_error > 0 else 0
        
        # F-table values
        f_table_5_factor_a = stats.f.ppf(0.95, df_factor_a, df_error)
        f_table_1_factor_a = stats.f.ppf(0.99, df_factor_a, df_error)
        f_table_5_factor_b = stats.f.ppf(0.95, df_factor_b, df_error)
        f_table_1_factor_b = stats.f.ppf(0.99, df_factor_b, df_error)
        f_table_5_interaction = stats.f.ppf(0.95, df_interaction, df_error)
        f_table_1_interaction = stats.f.ppf(0.99, df_interaction, df_error)
        
        # Significance levels
        alpha_a, sig_msg_a = get_significance_level(f_factor_a, f_table_5_factor_a, f_table_1_factor_a)
        alpha_b, sig_msg_b = get_significance_level(f_factor_b, f_table_5_factor_b, f_table_1_factor_b)
        alpha_int, sig_msg_int = get_significance_level(f_interaction, f_table_5_interaction, f_table_1_interaction)
        
        # Koefisien Keragaman
        cv = (np.sqrt(ms_error) / grand_mean * 100) if grand_mean != 0 else 0
        
        # Create ANOVA table
        anova_table = {
            'Source': ['Factor A', 'Factor B', 'Interaction', 'Error', 'Total'],
            'DF': [df_factor_a, df_factor_b, df_interaction, df_error, df_total],
            'SS': [round(ss_factor_a, 3), round(ss_factor_b, 3), round(ss_interaction, 3), 
                  round(ss_error, 3), round(ss_total, 3)],
            'MS': [round(ms_factor_a, 3), round(ms_factor_b, 3), round(ms_interaction, 3),
                  round(ms_error, 3), '-'],
            'F': [round(f_factor_a, 3), round(f_factor_b, 3), round(f_interaction, 3), '-', '-'],
            'Ftab5': [round(f_table_5_factor_a, 3), round(f_table_5_factor_b, 3), 
                     round(f_table_5_interaction, 3), '-', '-'],
            'Ftab1': [round(f_table_1_factor_a, 3), round(f_table_1_factor_b, 3),
                     round(f_table_1_interaction, 3), '-', '-']
        }
        
        # Convert all numeric values to float and handle NaN
        anova_table = {
            k: [float(v) if isinstance(v, (int, float)) and not isinstance(v, str) and not pd.isna(v) else v 
                for v in vals]
            for k, vals in anova_table.items()
        }
        
        result = {
            'summary': summary_dict,
            'anova': anova_table,
            'cv': round(float(cv), 2),
            'significance': {
                'factor_a': sig_msg_a,
                'factor_b': sig_msg_b,
                'interaction': sig_msg_int
            }
        }
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        stack_trace = traceback.format_exc()
        return {'error': f"{error_msg}\n{stack_trace}"}
