import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes
from scipy.interpolate import splprep, splev
from matplotlib.lines import Line2D
import re

try:
    import pyefd
except ImportError:
    pyefd = None

try:
    from IPython.display import display
except ImportError:
    def display(obj): print(obj)

__version__ = "0.1.0"

class MorphoGet:
    # =========================================================================
    # BLOCK 0: DICTIONARIES
    # =========================================================================
    KEY_LANDMARKS = ['Basal', 'Apex', 'Costa', 'Tornal', 'TaperCosta', 'TaperMargin']
    AVAILABLE_METRICS = [
        'MA_l', 'AR_l', 'AR_s', 'SA_l', 'SA_a', 'Taper_K', 'MA_s', 'Taper_l', 
        'Proximal_l', 'Distal_l', 'Costa_l', 'Tornal_l', 'L', 'C', 'Taper', 
        'Costa_A', 'Tornal_A'
    ]
    
    _BASE_FA_METRICS = [
        'MA_l', 'MA_s', 'AR_l', 'AR_s', 'SA_l', 'SA_a', 'Taper_l', 'Taper_K'
    ]
    
    _DEFAULT_FA_PARAMS = {
        'Basal': 1, 'Apex': 5, 'Costa': 3, 'Tornal': 7,
        'TaperCosta': None, 'TaperMargin': None
    }
    
    _BIPLOT_PRESETS = {
        'ma_biplot': ['Proximal_l', 'Distal_l'],
        'sa_biplot': ['Costa_l', 'Tornal_l'],
        'taper_biplot': ['Taper', 'C'],
        'sa_a_biplot': ['Costa_A', 'Tornal_A']
    }

    # =========================================================================
    # BLOCK 1: INITIALIZATION AND SETUP
    # =========================================================================
    def __init__(self, df, is_pre_aligned=False, fa_params=None, **fa_overrides):
        self.raw_df = df.copy()
        self.current_df = df.copy()
        self.is_pre_aligned = is_pre_aligned
        self.xy_cols = []
        self.labels = []
        self.neutralized_df = None
        self.lengths_df = None
        
        self.fa_params = self._DEFAULT_FA_PARAMS.copy()
        self.use_taper = False

        self.fa_metrics = list(self._BASE_FA_METRICS)
        
        if fa_params is not None: self.fa_params.update(fa_params)
        if fa_overrides: self.set_fa_params(**fa_overrides)

        # Хранилище для EFA
        self.efa_coeffs_df = None
        self.efa_order = None
        self.efa_normalize = None
        self.efa_locus = None
        self.efa_pca = None
        self.efa_pc_scores = None

    def print_fa_params(self):
        print("\nAvailable landmarks:", self.KEY_LANDMARKS)
        print("Current FA parameters:")
        for name, pt_id in self.fa_params.items():
            status = f"point {pt_id}" if pt_id is not None else "❌ NOT SET"
            print(f"  {name:12} : {status}")
        return self

    def set_fa_params(self, **kwargs):
        valid_keys = set(self.KEY_LANDMARKS)
        for key, value in kwargs.items():
            if key in valid_keys:
                self.fa_params[key] = value
                print(f"✅ '{key}' -> point {value}")
            else:
                print(f"⚠️ Key '{key}' ignored. Available: {valid_keys}")
        
        if self.fa_params.get('TaperCosta') is not None and self.fa_params.get('TaperMargin') is not None:
            if not self.use_taper:
                self.use_taper = True
                print("🔄 Taper automatically enabled.")
        else:
            if self.use_taper:
                self.use_taper = False
                print("🔄 Taper automatically disabled.")
        return self

    def filter_data(self, column, value):
        self.current_df[column] = self.current_df[column].astype(str).str.upper().str.strip()
        self.current_df = self.current_df[self.current_df[column] == str(value).upper()].copy()
        if self.neutralized_df is not None:
            self.neutralized_df = self.neutralized_df[self.neutralized_df[column] == str(value).upper()].copy()
        print(f"Filter {column}={value}. Remaining: {len(self.current_df)}")
        return self

    def set_landmarks(self, cols=None):
        if cols is not None:
            self.xy_cols = [c for c in cols if c in self.current_df.columns]
        else:
            pat_x = re.compile(r"^[Xx](\d+)$")
            pat_y = re.compile(r"^[Yy](\d+)$")
            x_map = {int(m.group(1)): c for c in self.current_df.columns if (m := pat_x.match(str(c)))}
            y_map = {int(m.group(1)): c for c in self.current_df.columns if (m := pat_y.match(str(c)))}
            self.xy_cols = []
            for i in sorted(set(x_map) & set(y_map)):
                self.xy_cols.extend([x_map[i], y_map[i]])
                
        for col in self.xy_cols:
            self.current_df[col] = pd.to_numeric(self.current_df[col], errors='coerce').astype('float64')
            
        self.labels = [re.search(r'\d+', c).group() for c in self.xy_cols if c.upper().startswith('X')]
        print(f"Selected {len(self.xy_cols)//2} points.")
        return self

    # =========================================================================
    # BLOCK 2: GPA AND NEUTRALIZATION
    # =========================================================================
    def run_gpa(self):
        if self.is_pre_aligned:
            print("⚠️ Data marked as pre-aligned. GPA skipped.")
            valid_idx = self.current_df[self.xy_cols].dropna().index
            self.consensus_shape = self.current_df.loc[valid_idx, self.xy_cols].mean().values.reshape(-1, 2)
            return self
        valid_idx = self.current_df[self.xy_cols].dropna().index
        coords = self.current_df.loc[valid_idx, self.xy_cols].values
        mean_shape = np.mean(coords, axis=0)
        for _ in range(10):
            aligned = []
            for shape in coords:
                _, mtx, _ = procrustes(mean_shape.reshape(-1, 2), shape.reshape(-1, 2))
                aligned.append(mtx.flatten())
            aligned = np.array(aligned)
            mean_shape = np.mean(aligned, axis=0)
        self.current_df.loc[valid_idx, self.xy_cols] = aligned
        self.consensus_shape = mean_shape.reshape(-1, 2)
        self.neutralized_df = None
        print("GPA completed.")
        return self

    def neutralize_pcs(self, components_to_remove=[0]):
        if self.is_pre_aligned:
            print("⚠️ Data marked as pre-aligned. Neutralization skipped.")
            self.neutralized_df = self.current_df.copy()
            return self
        valid_idx = self.current_df[self.xy_cols].dropna().index
        X = self.current_df.loc[valid_idx, self.xy_cols].values
        pca = PCA(n_components=min(X.shape[0], X.shape[1]))
        X_transformed = pca.fit_transform(X)
        for idx in components_to_remove:
            print(f"Removed PC{idx+1} ({pca.explained_variance_ratio_[idx]*100:.1f}%)")
            X_transformed[:, idx] = 0
        self.neutralized_df = self.current_df.copy()
        self.neutralized_df.loc[valid_idx, self.xy_cols] = pca.inverse_transform(X_transformed)
        return self

    def add_metrics(self, metrics_df):
        for col in metrics_df.columns:
            self.current_df[col] = metrics_df[col]
            if self.neutralized_df is not None:
                self.neutralized_df[col] = metrics_df[col]
        return self

    # =========================================================================
    # BLOCK 3.1: FA CALCULATION
    # =========================================================================
    def _get_col_name(self, df, axis, pt_num):
        target = f"{axis.upper()}{pt_num}"
        for col in df.columns:
            if str(col).upper() == target: return col
        raise KeyError(f"Column {target} not found.")

    def calculate_fa_metrics(self, ma_l_comp=1, sa_l_comp=1, sa_a_comp=1, taper_l_comp=1):
        source_df = self.raw_df.loc[self.current_df.index].copy()
        p = self.fa_params
        
        if self.is_pre_aligned:
            print("⚠️ WARNING: FA calculation uses pre-aligned data (Procrustes units, not mm).")

        FA_df = pd.DataFrame({
            f'{k}_{ax}': source_df[self._get_col_name(source_df, ax.lower(), p[k])]
            for k in ['Basal', 'Apex', 'Costa', 'Tornal'] for ax in ['X', 'Y']
        }).dropna()

        if self.use_taper:
            FA_df = FA_df.join(pd.DataFrame({
                f'{k}_{ax}': source_df[self._get_col_name(source_df, ax.lower(), p[k])]
                for k in ['TaperCosta', 'TaperMargin'] for ax in ['X', 'Y']
            })).dropna()

        A1 = FA_df['Basal_Y'] - FA_df['Apex_Y']; B1 = FA_df['Apex_X'] - FA_df['Basal_X']; C1 = A1 * FA_df['Basal_X'] + B1 * FA_df['Basal_Y']
        A2 = FA_df['Costa_Y'] - FA_df['Tornal_Y']; B2 = FA_df['Tornal_X'] - FA_df['Costa_X']; C2 = A2 * FA_df['Costa_X'] + B2 * FA_df['Costa_Y']
        den = A1 * B2 - A2 * B1
        x0 = ((B2 * C1 - B1 * C2) / den).replace([np.inf, -np.inf], np.nan)
        y0 = ((A1 * C2 - A2 * C1) / den).replace([np.inf, -np.inf], np.nan)

        L = np.sqrt((FA_df['Basal_X'] - FA_df['Apex_X'])**2 + (FA_df['Basal_Y'] - FA_df['Apex_Y'])**2)
        C = np.sqrt((FA_df['Tornal_X'] - FA_df['Costa_X'])**2 + (FA_df['Tornal_Y'] - FA_df['Costa_Y'])**2)
        Distal_l = np.sqrt((x0 - FA_df['Apex_X'])**2 + (y0 - FA_df['Apex_Y'])**2)
        Proximal_l = np.sqrt((FA_df['Basal_X'] - x0)**2 + (FA_df['Basal_Y'] - y0)**2)
        Costa_l = np.sqrt((x0 - FA_df['Costa_X'])**2 + (y0 - FA_df['Costa_Y'])**2)
        Tornal_l = np.sqrt((FA_df['Tornal_X'] - x0)**2 + (FA_df['Tornal_Y'] - y0)**2)
        
        Taper = pd.Series(np.nan, index=FA_df.index)
        if self.use_taper:
            Taper = np.sqrt((FA_df['TaperCosta_X'] - FA_df['TaperMargin_X'])**2 + (FA_df['TaperCosta_Y'] - FA_df['TaperMargin_Y'])**2)

        vL_x = FA_df['Apex_X'] - FA_df['Basal_X']; vL_y = FA_df['Apex_Y'] - FA_df['Basal_Y']
        vC_x = FA_df['Costa_X'] - FA_df['Basal_X']; vC_y = FA_df['Costa_Y'] - FA_df['Basal_Y']
        vT_x = FA_df['Tornal_X'] - FA_df['Basal_X']; vT_y = FA_df['Tornal_Y'] - FA_df['Basal_Y']
        BC_len = np.sqrt(vC_x**2 + vC_y**2); BT_len = np.sqrt(vT_x**2 + vT_y**2)
        Costa_A = np.arccos(np.clip((vL_x * vC_x + vL_y * vC_y) / (L * BC_len), -1.0, 1.0))
        Tornal_A = np.arccos(np.clip((vL_x * vT_x + vL_y * vT_y) / (L * BT_len), -1.0, 1.0))

        col_names = self.xy_cols
        id_to_idx = {int(re.search(r'\d+', c).group()): col_names.index(c) // 2 for c in col_names if c.upper().startswith('X')}
        Proximal_s = pd.Series(np.nan, index=FA_df.index); Distal_s = pd.Series(np.nan, index=FA_df.index)
        S_taper = pd.Series(np.nan, index=FA_df.index); MA_s = pd.Series(np.nan, index=FA_df.index); Taper_K = pd.Series(np.nan, index=FA_df.index); S = Proximal_s + Distal_s

        if all(k in id_to_idx for k in [p['Basal'], p['Apex'], p['Costa'], p['Tornal']]):
            idx_basal, idx_apex, idx_costa, idx_tornal = id_to_idx[p['Basal']], id_to_idx[p['Apex']], id_to_idx[p['Costa']], id_to_idx[p['Tornal']]
            n_points = len(id_to_idx)
            
            arc1 = list(range(idx_costa, idx_tornal + 1)) if idx_costa <= idx_tornal else list(range(idx_costa, n_points)) + list(range(0, idx_tornal + 1))
            arc2 = list(range(idx_tornal, n_points)) + list(range(0, idx_costa + 1)) if idx_costa <= idx_tornal else list(range(idx_tornal, idx_costa + 1))
            prox_arc = arc1 if idx_basal in arc1 else arc2; dist_arc = arc2 if idx_basal in arc1 else arc1
            
            contour_coords = source_df.loc[FA_df.index, self.xy_cols].values.reshape(-1, n_points, 2)
            p_inter_np = np.column_stack((x0, y0)).reshape(-1, 1, 2)
            
            def calc_poly_area(arc_idx, close=False):
                pts = contour_coords[:, arc_idx, :]
                if close: pts = np.concatenate([pts, p_inter_np], axis=1)
                x, y = pts[:, :, 0], pts[:, :, 1]
                return 0.5 * np.abs(np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1))
            
            Proximal_s = pd.Series(calc_poly_area(prox_arc, close=True), index=FA_df.index)
            Distal_s = pd.Series(calc_poly_area(dist_arc, close=True), index=FA_df.index)
            S = Proximal_s + Distal_s
            MA_s = Proximal_s / S

            arc_ca = list(range(idx_costa, idx_apex + 1)) if idx_costa <= idx_apex else list(range(idx_costa, n_points)) + list(range(0, idx_apex + 1))
            arc_at = list(range(idx_apex, idx_tornal + 1)) if idx_apex <= idx_tornal else list(range(idx_apex, n_points)) + list(range(0, idx_tornal + 1))
            S_taper = pd.Series(calc_poly_area(arc_ca + arc_at[1:]), index=FA_df.index)
            
            A_ct = FA_df['Costa_Y'] - FA_df['Tornal_Y']; B_ct = FA_df['Tornal_X'] - FA_df['Costa_X']
            C_ct = FA_df['Costa_X'] * FA_df['Tornal_Y'] - FA_df['Tornal_X'] * FA_df['Costa_Y']
            denom_ct = np.sqrt(A_ct**2 + B_ct**2)
            H_real = np.abs(A_ct * FA_df['Apex_X'] + B_ct * FA_df['Apex_Y'] + C_ct).div(denom_ct).replace([np.inf, -np.inf], np.nan)
            S_max_triangle = 0.5 * C * H_real; K_raw = S_taper.div(S_max_triangle).replace([np.inf, -np.inf], np.nan)
            Taper_K = (np.pi / 2 - K_raw) / (np.pi / 2 - 1)

        self.lengths_df = pd.DataFrame({
            'Distal_l': Distal_l, 'L': L, 'Proximal_l': Proximal_l, 'Costa_l': Costa_l, 'Tornal_l': Tornal_l, "C": C, 'Taper': Taper,
            'MA_s': MA_s, 'Taper_K': Taper_K, 'Costa_A': Costa_A, 'Tornal_A': Tornal_A
        }, index=FA_df.index)

        pca1 = PCA()
        idx_ma = self.lengths_df[['Proximal_l', 'Distal_l']].dropna().index
        self.lengths_df.loc[idx_ma, '2MA'] = pca1.fit_transform(self.lengths_df.loc[idx_ma, ['Proximal_l', 'Distal_l']])[:, ma_l_comp]
        
        idx_sa = self.lengths_df[['Costa_l', 'Tornal_l']].dropna().index
        self.lengths_df.loc[idx_sa, 'SA_l'] = pca1.fit_transform(self.lengths_df.loc[idx_sa, ['Costa_l', 'Tornal_l']])[:, sa_l_comp]
        
        idx_sa_a = self.lengths_df[['Costa_A', 'Tornal_A']].dropna().index
        self.lengths_df.loc[idx_sa_a, 'SA_a'] = pca1.fit_transform(self.lengths_df.loc[idx_sa_a, ['Costa_A', 'Tornal_A']])[:, sa_a_comp]
        
        if self.use_taper:
            idx_t = self.lengths_df[['Taper', "C"]].dropna().index
            self.lengths_df.loc[idx_t, 'Taper_l'] = pca1.fit_transform(self.lengths_df.loc[idx_t, ['Taper', "C"]])[:, taper_l_comp]
        
        metrics_to_add = self.lengths_df[['2MA', 'SA_l', 'MA_s', 'Taper_K', 'SA_a']].rename(columns={'2MA': 'MA_l'})
        metrics_to_add['AR_l'] = L / C
        metrics_to_add['AR_s'] = (L ** 2) / S
        if self.use_taper:
            metrics_to_add['Taper_l'] = self.lengths_df['Taper_l']
            
        self.add_metrics(metrics_to_add)
        
        calculated_metrics = [m for m in self.fa_metrics if m in metrics_to_add.columns]
        comp_info = f"(MA_l={ma_l_comp}, SA_l={sa_l_comp}, SA_a={sa_a_comp}"
        if self.use_taper: comp_info += f", Taper_l={taper_l_comp}"
        comp_info += ")"
        msg = f"✅ FA calculated {comp_info}: {', '.join(calculated_metrics)}"
        print(msg)
        return self

    # =========================================================================
    # BLOCK 3.2: ELLIPTICAL FOURIER ANALYSIS
    # =========================================================================
    def calculate_efa(self, order=10, normalize=False, use_neutralized=False):
        if pyefd is None:
            raise ImportError("pyefd library is not installed. Run: pip install pyefd")
        if not self.xy_cols:
            raise ValueError("Call set_landmarks() first")
            
        source_df = self.current_df
        if use_neutralized and self.neutralized_df is not None:
            source_df = self.neutralized_df
            
        valid_idx = source_df[self.xy_cols].dropna().index
        n_points = len(self.xy_cols) // 2
        coords = source_df.loc[valid_idx, self.xy_cols].values.reshape(-1, n_points, 2)
        
        all_coeffs = []
        loci = []
        for contour in coords:
            coeffs = pyefd.elliptic_fourier_descriptors(contour, order=order, normalize=normalize)
            all_coeffs.append(coeffs.flatten())
            loci.append(pyefd.calculate_dc_coefficients(contour))
            
        coeffs_arr = np.array(all_coeffs)
        loci_arr = np.array(loci)
        
        col_names = [f"EFD_H{h+1}_{c}" for h in range(order) for c in ['A', 'B', 'C', 'D']]
        self.efa_coeffs_df = pd.DataFrame(coeffs_arr, index=valid_idx, columns=col_names)
        self.efa_locus = loci_arr
        self.efa_order = order
        self.efa_normalize = normalize
        self.efa_pca = None
        
        print(f"✅ EFA calculated: {len(valid_idx)} contours, order={order}, normalize={normalize}")
        return self

    def run_efa_pca(self, n_components=5):
        if self.efa_coeffs_df is None:
            raise ValueError("Call calculate_efa() first")
            
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(self.efa_coeffs_df.values)
        
        pc_cols = [f'EFA_PC{i+1}' for i in range(n_components)]
        self.efa_pc_scores = pd.DataFrame(scores, index=self.efa_coeffs_df.index, columns=pc_cols)
        self.efa_pca = pca
        
        print(f"✅ PCA on EFA completed. Explained variance ratio: {pca.explained_variance_ratio_.round(3)}")
        return self

    def _reconstruct_efa_shape(self, coeffs_flat, locus, n_points=200):
        coeffs = coeffs_flat.reshape(self.efa_order, 4)
        t = np.linspace(0, 2 * np.pi, n_points)
        x_recon = np.full(n_points, locus[0])
        y_recon = np.full(n_points, locus[1])
        for n in range(self.efa_order):
            A, B, C, D = coeffs[n]
            harmonic = n + 1
            x_recon += A * np.cos(harmonic * t) + B * np.sin(harmonic * t)
            y_recon += C * np.cos(harmonic * t) + D * np.sin(harmonic * t)
        return np.column_stack((x_recon, y_recon))

    # =========================================================================
    # BLOCK 4: CORRELATIONS
    # =========================================================================
    def show_correlations(self, fa_metrics=None, n_pcs=4, use_neutralized=True):
        if fa_metrics is None: 
            fa_metrics = [m for m in self.fa_metrics if m != 'Taper_l' or self.use_taper]
            
        valid_cols = [c for c in fa_metrics if c in self.current_df.columns and not self.current_df[c].isna().all()]
        missing = set(fa_metrics) - set(valid_cols)
        if missing: 
            print(f"⚠️ Metrics {missing} skipped or not calculated. (Did you call calculate_fa_metrics()?)")
        if not valid_cols: 
            print("❌ Nothing to correlate."); 
            return

        source_df = self.neutralized_df if use_neutralized and self.neutralized_df is not None else self.current_df
        valid_idx = source_df[self.xy_cols].dropna().index
        X_shape = source_df.loc[valid_idx, self.xy_cols].values
        
        max_possible_pcs = min(X_shape.shape[0], X_shape.shape[1])
        actual_n_pcs = min(n_pcs, max_possible_pcs)
        
        pc_scores = PCA(n_components=actual_n_pcs).fit_transform(X_shape)
        pc_cols = [f'Shape_PC{i+1}' for i in range(actual_n_pcs)]
        
        pc_df = pd.DataFrame(pc_scores, columns=pc_cols, index=valid_idx)
        combined_df = pd.concat([pc_df, source_df.loc[valid_idx, valid_cols]], axis=1).dropna()
        if combined_df.empty: print("⚠️ Insufficient data."); return
            
        corr_matrix = combined_df.corr(method='pearson').loc[pc_cols, valid_cols]
        try: 
            display(corr_matrix.round(3).style.background_gradient(cmap='Reds', axis=None, vmin=0, vmax=1, gmap=corr_matrix.abs()).format("{:.3f}"))
        except (TypeError, AttributeError): 
            try:
                display(corr_matrix.round(3).style.background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1).format("{:.3f}"))
            except AttributeError:
                print("⚠️ Для цветного вывода установите jinja2: pip install jinja2")
                display(corr_matrix.round(3))

    def show_efa_correlations(self, fa_metrics=None, n_efa_pcs=4):
        if self.efa_pc_scores is None:
            raise ValueError("Call calculate_efa() and run_efa_pca() first")
            
        if fa_metrics is None: 
            fa_metrics = [m for m in self.fa_metrics if m != 'Taper_l' or self.use_taper]
            
        valid_cols = [c for c in fa_metrics if c in self.current_df.columns and not self.current_df[c].isna().all()]
        missing = set(fa_metrics) - set(valid_cols)
        if missing: 
            print(f"⚠️ Metrics {missing} skipped or not calculated. (Did you call calculate_fa_metrics()?)")
        if not valid_cols: 
            print("❌ Nothing to correlate."); 
            return

        efa_pc_cols = [f'EFA_PC{i+1}' for i in range(min(n_efa_pcs, len(self.efa_pc_scores.columns)))]
        
        common_idx = self.efa_pc_scores.index.intersection(self.current_df.dropna(subset=valid_cols).index)
        if common_idx.empty:
            print("❌ No overlapping data between EFA PCs and metrics."); return
            
        efa_df = self.efa_pc_scores.loc[common_idx, efa_pc_cols]
        metrics_df = self.current_df.loc[common_idx, valid_cols]
        
        combined_df = pd.concat([efa_df, metrics_df], axis=1)
        corr_matrix = combined_df.corr(method='pearson').loc[efa_pc_cols, valid_cols]
        
        try: 
            display(corr_matrix.round(3).style.background_gradient(cmap='Reds', axis=None, vmin=0, vmax=1, gmap=corr_matrix.abs()).format("{:.3f}"))
        except (TypeError, AttributeError): 
            try:
                display(corr_matrix.round(3).style.background_gradient(cmap='RdBu_r', axis=None, vmin=-1, vmax=1).format("{:.3f}"))
            except AttributeError:
                print("⚠️ Install jinja2 for colored output: pip install jinja2")
                display(corr_matrix.round(3))

    # =========================================================================
    # BLOCK 5: MORPHOSPACE VISUALIZATION
    # =========================================================================
    def plot_biplot(self, *presets, title=None, figsize=(5, 4)):
        if self.lengths_df is None: raise ValueError("Call calculate_fa_metrics() first")
        if not presets:
            presets_to_plot = list(self._BIPLOT_PRESETS.keys())
        elif len(presets) == 1 and isinstance(presets[0], list):
            presets_to_plot = presets[0]
        else:
            presets_to_plot = list(presets)
            
        for preset in presets_to_plot:
            if preset not in self._BIPLOT_PRESETS: 
                raise ValueError(f"Preset '{preset}' not found. Available: {list(self._BIPLOT_PRESETS.keys())}")
            if preset == 'taper_biplot' and not self.use_taper: 
                print(f"⚠️ Skipping '{preset}': Taper is disabled.")
                continue
            columns = self._BIPLOT_PRESETS[preset]
            data = self.lengths_df[columns].dropna().copy()
            if data.empty:
                continue
            try:
                Xz = StandardScaler().fit_transform(data.values)
                pca = PCA(n_components=2, random_state=42)
                scores = pca.fit_transform(Xz)
                loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
            except Exception as e:
                continue
            
            plt.figure(figsize=figsize)
            plt.scatter(scores[:, 0], scores[:, 1], s=22, alpha=0.55)
            arrow_scale = 2.5
            for i, name in enumerate(columns):
                x, y = loadings[i, 0], loadings[i, 1]
                plt.arrow(0, 0, arrow_scale * x, arrow_scale * y, color='red', width=0.01, head_width=0.10, length_includes_head=True)
                plt.text(arrow_scale * x * 1.08, arrow_scale * y * 1.08, name, color='red', fontsize=11)
            evr = pca.explained_variance_ratio_ * 100
            plt.axhline(0, color='gray', lw=1); plt.axvline(0, color='gray', lw=1)
            plt.xlabel(f'PC1 ({evr[0]:.1f}%)'); plt.ylabel(f'PC2 ({evr[1]:.1f}%)')
            plot_title = title if (title and len(presets_to_plot) == 1) else preset
            plt.title(plot_title)
            plt.tight_layout(); plt.show()

    def plot_consensus(self, use_neutralized=True, figsize=(8, 8), smooth=True, flip_x=False, flip_y=False, show_lines=True):
        source_df = self.neutralized_df if use_neutralized and self.neutralized_df is not None else self.current_df
        X = source_df[self.xy_cols].dropna().values
        mean_shape = self._flip_coords(np.mean(X, axis=0).reshape(-1, 2), flip_x, flip_y)
        plt.figure(figsize=figsize)
        if show_lines:
            pts = self._smooth_contour(mean_shape) if smooth else self._close_contour(mean_shape)
            plt.plot(pts[:, 0], pts[:, 1], 'k-', lw=2.5 if smooth else 1.5)
        plt.scatter(mean_shape[:, 0], mean_shape[:, 1], c='red', s=40, zorder=5)
        for i, (x, y) in enumerate(mean_shape): plt.text(x, y, str(self.labels[i]), fontsize=9, ha='center', va='bottom', color='red')
        plt.title("Mean shape (Consensus)"); plt.gca().set_aspect('equal'); plt.axis('off'); plt.show()

    def _flip_coords(self, coords_2d, flip_x=False, flip_y=False):
        pts = np.array(coords_2d).copy()
        if flip_x: pts[:, 0] = -pts[:, 0]
        if flip_y: pts[:, 1] = -pts[:, 1]
        return pts

    def _smooth_contour(self, coords, n_points=200):
        if len(coords) < 4: return np.column_stack((coords[:, 0], coords[:, 1]))
        pts = np.vstack([coords, coords[0]]) if not np.allclose(coords[0], coords[-1]) else coords
        try:
            tck, u = splprep([pts[:, 0], pts[:, 1]], s=0, k=3, per=True)
            return np.column_stack(splev(np.linspace(0, 1, n_points), tck))
        except Exception: return pts

    def _close_contour(self, coords):
        return np.vstack([coords, coords[0]]) if not np.allclose(coords[0], coords[-1]) else coords

    def _plot_morphospace_grid(self, shapes_data, titles, n_cols=2, z_ext=3.0, smooth=False, flip_x=False, flip_y=False, figsize=None):
        colors = ['silver', 'dimgray', 'black']
        n_rows = int(np.ceil(len(shapes_data) / n_cols))
        if figsize is None:
            figsize = (7.5*n_cols, 3.0*n_rows)
            
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.ravel()
        
        for i, (shape_raw, title) in enumerate(zip(shapes_data, titles)):
            if i >= len(axes): break
            ax = axes[i]
            for j, color in enumerate(colors):
                shape = self._flip_coords(shape_raw[j].reshape(-1, 2), flip_x, flip_y)
                pts = self._smooth_contour(shape) if smooth else self._close_contour(shape)
                ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.8)
            shape_mid = self._flip_coords(shape_raw[1].reshape(-1, 2), flip_x, flip_y)
            ax.scatter(shape_mid[:, 0], shape_mid[:, 1], s=12, color=colors[1], alpha=0.7)
            ax.set_title(title, fontsize=12); ax.set_aspect('equal', adjustable='box'); ax.axis('off')
            
        for j in range(len(shapes_data), len(axes)): axes[j].set_visible(False)
        fig.legend(handles=[
            Line2D([0], [0], color=colors[0], lw=3, label=rf"$-{z_ext:.0f}\sigma$"),
            Line2D([0], [0], color=colors[1], lw=3, label="mean"),
            Line2D([0], [0], color=colors[2], lw=3, label=rf"$+{z_ext:.0f}\sigma$")
        ], loc="upper center", ncol=3, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.92]); plt.show()

    def plot_pca(self, pc_indices=[0, 1, 2, 3], z_ext=3.0, n_cols=2, use_neutralized=True, smooth=False, 
                 flip_x=False, flip_y=False, invert_sigma=[], figsize=None):
        source_df = self.neutralized_df if use_neutralized and self.neutralized_df is not None else self.current_df
        X = source_df[self.xy_cols].dropna().values; pca = PCA().fit(X); mean_vec = pca.mean_
        shapes, titles = [], []
        for idx in pc_indices:
            comp = np.sqrt(pca.explained_variance_[idx]) * pca.components_[idx]
            neg, pos = mean_vec - z_ext * comp, mean_vec + z_ext * comp
            shapes.append((pos, mean_vec, neg) if idx in invert_sigma else (neg, mean_vec, pos))
            titles.append(f"PC{idx+1} ({pca.explained_variance_ratio_[idx]*100:.1f}%)")
        self._plot_morphospace_grid(shapes, titles, n_cols, z_ext, smooth, flip_x, flip_y, figsize)

    def plot_efa_pca(self, pc_indices=[0, 1, 2, 3], z_ext=3.0, n_cols=2, n_points=200, 
                     flip_x=False, flip_y=False, invert_sigma=[], figsize=None):
        if pyefd is None:
            raise ImportError("pyefd library is not installed. Run: pip install pyefd")
        if self.efa_pca is None or self.efa_pc_scores is None:
            raise ValueError("Call calculate_efa() and run_efa_pca() first")
            
        mean_coeffs = self.efa_pca.mean_
        mean_locus = self.efa_locus.mean(axis=0)
        
        valid_pc_indices = [idx for idx in pc_indices if idx < self.efa_pca.n_components_]
        if len(valid_pc_indices) < len(pc_indices):
            print(f"⚠️ Requested PCs {pc_indices}, but only {self.efa_pca.n_components_} computed. Showing {valid_pc_indices}")
        
        shapes, titles = [], []
        for idx in valid_pc_indices:
            comp = np.sqrt(self.efa_pca.explained_variance_[idx]) * self.efa_pca.components_[idx]
            neg_coeffs = mean_coeffs - z_ext * comp
            pos_coeffs = mean_coeffs + z_ext * comp
            
            shape_neg = self._reconstruct_efa_shape(neg_coeffs, mean_locus, n_points)
            shape_mean = self._reconstruct_efa_shape(mean_coeffs, mean_locus, n_points)
            shape_pos = self._reconstruct_efa_shape(pos_coeffs, mean_locus, n_points)
            
            shapes.append((shape_pos.flatten(), shape_mean.flatten(), shape_neg.flatten()) if idx in invert_sigma 
                          else (shape_neg.flatten(), shape_mean.flatten(), shape_pos.flatten()))
            titles.append(f"EFA PC{idx+1} ({self.efa_pca.explained_variance_ratio_[idx]*100:.1f}%)")
            
        if shapes:
            self._plot_morphospace_grid(shapes, titles, n_cols, z_ext, smooth=False, flip_x=flip_x, flip_y=flip_y, figsize=figsize)

    def plot_regressions(self, fa_metrics=None, z_ext=3.0, n_cols=2, use_neutralized=True, smooth=True, 
                         flip_x=False, flip_y=False, invert_sigma=[], figsize=None):
        if fa_metrics is None:
            fa_metrics = [m for m in self.fa_metrics if m != 'Taper_l' or self.use_taper]
            
        source_df = self.neutralized_df if use_neutralized and self.neutralized_df is not None else self.current_df
        if any(c not in source_df.columns for c in fa_metrics): raise ValueError("Metrics not found")
        valid_data = source_df[self.xy_cols + fa_metrics].dropna()
        if len(valid_data) < 5: return
        X_shape = valid_data[self.xy_cols].values; Y_metrics = valid_data[fa_metrics].values
        shapes, titles = [], []
        for i, col in enumerate(fa_metrics):
            y = Y_metrics[:, i].reshape(-1, 1); y_mean, y_std = y.mean(), y.std()
            if y_std == 0: continue
            model = LinearRegression().fit((y - y_mean) / y_std, X_shape)
            intercept, slope = model.intercept_.flatten(), model.coef_.flatten()
            neg, pos = intercept - z_ext * slope, intercept + z_ext * slope
            shapes.append((pos, intercept, neg) if col in invert_sigma else (neg, intercept, pos))
            titles.append(col)
        self._plot_morphospace_grid(shapes, titles, n_cols, z_ext, smooth, flip_x, flip_y, figsize)

    # =========================================================================
    # BLOCK 6: SCHEMES 
    # =========================================================================
    def _get_fa_coords_and_intersection(self, source_df):
        p = self.fa_params
        if not self.xy_cols: raise ValueError("Call set_landmarks() first")
        col_names = self.xy_cols
        id_to_idx = {int(re.search(r'\d+', c).group()): col_names.index(c) // 2 for c in col_names if c.upper().startswith('X')}
        for key, pt_id in p.items():
            if pt_id is not None and pt_id not in id_to_idx: raise KeyError(f"Point {pt_id} ('{key}') not found.")
        
        coords = source_df[self.xy_cols].mean().values.reshape(-1, 2)
        def get_pt(pt_id): return coords[id_to_idx[pt_id]] if pt_id is not None else None
        
        basal, apex = get_pt(p['Basal']), get_pt(p['Apex'])
        costa, tornal = get_pt(p['Costa']), get_pt(p['Tornal'])
        taper_c, taper_t = get_pt(p['TaperCosta']), get_pt(p['TaperMargin'])
        
        A1 = basal[1] - apex[1]; B1 = apex[0] - basal[0]; C1_val = A1 * basal[0] + B1 * basal[1]
        A2 = costa[1] - tornal[1]; B2 = tornal[0] - costa[0]; C2_val = A2 * costa[0] + B2 * costa[1]
        det = A1 * B2 - A2 * B1
        if det == 0: return None
        p_inter = np.array([(B2 * C1_val - B1 * C2_val) / det, (A1 * C2_val - A2 * C1_val) / det])
        return coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter

    def _draw_fa_schema(self, coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter, mean_lengths, flip_x, flip_y, show_individual=False, source_df=None):
        coords = self._flip_coords(coords, flip_x, flip_y)
        if flip_x or flip_y:
            basal = self._flip_coords([basal], flip_x, flip_y)[0]
            apex = self._flip_coords([apex], flip_x, flip_y)[0]
            costa = self._flip_coords([costa], flip_x, flip_y)[0]
            tornal = self._flip_coords([tornal], flip_x, flip_y)[0]
            p_inter = self._flip_coords([p_inter], flip_x, flip_y)[0]
            if taper_c is not None: taper_c = self._flip_coords([taper_c], flip_x, flip_y)[0]
            if taper_t is not None: taper_t = self._flip_coords([taper_t], flip_x, flip_y)[0]

        OFFSET_VAL = 0.015
        p = self.fa_params
        
        if mean_lengths is None:
            mean_lengths = {
                'L': np.linalg.norm(apex - basal),
                'C': np.linalg.norm(costa - tornal),
                'Proximal_l': np.linalg.norm(p_inter - basal),
                'Distal_l': np.linalg.norm(apex - p_inter),
                'Costa_l': np.linalg.norm(costa - p_inter),
                'Tornal_l': np.linalg.norm(tornal - p_inter),
                'Taper': np.linalg.norm(taper_t - taper_c) if (taper_c is not None and taper_t is not None) else 0.0
            }
        
        def get_offset_points(p1, p2, offset_val):
            vec = np.array(p2) - np.array(p1); norm = np.linalg.norm(vec)
            if norm == 0: return p1, p2
            perp = np.array([-vec[1], vec[0]]) / norm
            return np.array(p1) + perp * offset_val, np.array(p2) + perp * offset_val
            
        def add_label_outward(pt, name):
            center = coords.mean(axis=0)
            vec = np.array(pt) - center
            dist = np.linalg.norm(vec)
            if dist == 0: return
            direction = vec / dist
            x_range = coords[:, 0].max() - coords[:, 0].min()
            y_range = coords[:, 1].max() - coords[:, 1].min()
            max_range = max(x_range, y_range)
            offset = max_range * 0.07
            text_x = pt[0] + direction[0] * offset
            text_y = pt[1] + direction[1] * offset
            ha = 'left' if direction[0] >= 0 else 'right'
            va = 'bottom' if direction[1] >= 0 else 'top'
            ax.text(text_x, text_y, name, fontsize=9, ha=ha, va=va, 
                    color='black', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

        fig = plt.figure(figsize=(12, 7)); ax = fig.add_axes([0.05, 0.1, 0.55, 0.8])
        arr = np.vstack([coords, coords[0]])
        try:
            tck, u = splprep([arr[:, 0], arr[:, 1]], s=0, k=3, per=True)
            ax.plot(*splev(np.linspace(0, 1, 200), tck), color='black', lw=2.5, zorder=1)
        except Exception: ax.plot(arr[:, 0], arr[:, 1], color='black', lw=2.5, zorder=1)
            
        ax.scatter(coords[:, 0], coords[:, 1], facecolors='white', edgecolors='black', s=40, zorder=4, linewidth=0.5)
        
        if show_individual and source_df is not None:
            fa_ids = [p[k] for k in ['Basal', 'Apex', 'Costa', 'Tornal', 'TaperCosta', 'TaperMargin'] if p[k] is not None]
            indiv_fa_pts = []
            for pt_id in fa_ids:
                try:
                    indiv_fa_pts.append(source_df[[self._get_col_name(source_df, 'x', pt_id), self._get_col_name(source_df, 'y', pt_id)]].dropna().values)
                except KeyError: pass
            if indiv_fa_pts:
                indiv_fa_pts = self._flip_coords(np.vstack(indiv_fa_pts), flip_x, flip_y)
                ax.scatter(indiv_fa_pts[:, 0], indiv_fa_pts[:, 1], facecolors='red', edgecolors='none', alpha=0.25, s=20, zorder=5)

        fa_pts = [pt for pt in [basal, apex, costa, tornal, taper_c, taper_t] if pt is not None]
        if fa_pts:
            ax.scatter(*np.array(fa_pts).T, facecolors='red', edgecolors='black', s=60, zorder=6, linewidth=1)

        add_label_outward(basal, f"Basal ({p['Basal']})")
        add_label_outward(apex, f"Apex ({p['Apex']})")
        add_label_outward(costa, f"Costa ({p['Costa']})")
        add_label_outward(tornal, f"Tornal ({p['Tornal']})")
        if taper_c is not None: add_label_outward(taper_c, f"T.Costa ({p['TaperCosta']})")
        if taper_t is not None: add_label_outward(taper_t, f"T.Margin ({p['TaperMargin']})")

        ax.scatter(*p_inter, facecolors='white', edgecolors='black', s=60, marker='o', zorder=3, linewidth=1)
        ax.plot([basal[0], apex[0]], [basal[1], apex[1]], color='#D62728', lw=2.0, zorder=2, ls='-')
        ax.plot([tornal[0], costa[0]], [tornal[1], costa[1]], color='#1F77B4', lw=2.0, zorder=2, ls='-')
        if taper_c is not None and taper_t is not None: 
            ax.plot([taper_c[0], taper_t[0]], [taper_c[1], taper_t[1]], color='#2CA02C', lw=2.0, zorder=2, ls='-')

        for p1, p2, off in [(basal, p_inter, OFFSET_VAL), (p_inter, apex, -OFFSET_VAL), (tornal, p_inter, OFFSET_VAL), (p_inter, costa, -OFFSET_VAL)]:
            pt1, pt2 = get_offset_points(p1, p2, off)
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color='black', lw=1.5, zorder=3, ls='--' if off > 0 else ':')

        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(coords[:, 0].min() - 0.15, coords[:, 0].max() + 0.15)
        ax.set_ylim(coords[:, 1].min() - 0.10, coords[:, 1].max() + 0.10)

        legend_x = 0.65
        current_y = 0.90
        step = 0.04
        
        fig.text(legend_x, current_y, "Absolute dimensions:", fontsize=12, fontweight='bold', va='center')
        current_y -= step

        main_lines = [('L', 'L', '#D62728', '-'), ('C', 'C', '#1F77B4', '-')]
        if taper_c is not None: main_lines.append(('Taper', 'Taper', '#2CA02C', '-'))
        
        for name, col, color, style in main_lines:
            fig.add_artist(Line2D([legend_x, legend_x + 0.03], [current_y, current_y], color=color, lw=3, ls=style, transform=fig.transFigure))
            fig.text(legend_x + 0.05, current_y, f"{name} = {mean_lengths[col]:.2f}", fontsize=11, va='center')
            current_y -= step

        current_y -= step * 0.5 
        fig.text(legend_x, current_y, "Segments:", fontsize=12, fontweight='bold', va='center')
        current_y -= step
        
        for name, col, style in [('Proximal', 'Proximal_l', '--'), ('Distal', 'Distal_l', ':'), ('Tornal', 'Tornal_l', '--'), ('Costa', 'Costa_l', ':')]:
            fig.add_artist(Line2D([legend_x, legend_x + 0.03], [current_y, current_y], color='black', lw=2.5, ls=style, transform=fig.transFigure))
            fig.text(legend_x + 0.05, current_y, f"{name} = {mean_lengths[col]:.2f}", fontsize=11, va='center')
            current_y -= step
            
        plt.show()

    def plot_schematic(self, use_neutralized=True, figsize=(12, 7), flip_x=False, flip_y=False):
        source_df = self.neutralized_df if use_neutralized and self.neutralized_df is not None else self.current_df
        res = self._get_fa_coords_and_intersection(source_df)
        if res is None: return
        coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter = res
        mean_lengths = self.lengths_df.mean() if self.lengths_df is not None else None
        self._draw_fa_schema(coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter, mean_lengths, flip_x, flip_y)

    def plot_fa_preview(self, figsize=(12, 7), flip_x=False, flip_y=False):
        source_df = self.raw_df.loc[self.current_df.index]
        res = self._get_fa_coords_and_intersection(source_df)
        if res is None: return
        coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter = res
        mean_lengths = self.lengths_df.mean() if self.lengths_df is not None else None
        self._draw_fa_schema(coords, basal, apex, costa, tornal, taper_c, taper_t, p_inter, mean_lengths, flip_x, flip_y, show_individual=True, source_df=source_df)