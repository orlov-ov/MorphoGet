
# Morpho**GET**

![Version](https://img.shields.io/badge/version-0.1.0-blue)
[![License: AGPL-3.0](https://img.shields.io/badge/Code_License-AGPL_3.0-red)](https://www.gnu.org/licenses/agpl-3.0.html)
[![License: CC0-1.0](https://img.shields.io/badge/Data_License-CC0_1.0-lightgrey)](http://creativecommons.org/publicdomain/zero/1.0/)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

**MorphoGet** is a Python library (class) for morphometric analysis of insect wing shape. The class enables automatic extraction of the most informative flight activity (FA) indicators from geometric landmarks (XY-coordinates) of digitized objects. Performs Generalized Procrustes Analysis (GPA), Elliptical Fourier Analysis (EFA), correlations and visualizes morphospace for comparison and establishment.

## Key Features

- **Flight Activity Metrics Calculation:** Computes aspect ratio, second moment of area, sweep angle, and wing taper from 4–6 homologous landmarks.
- **Contour Processing:** Automatic shape alignment [Generalized Procrustes Analysis (GPA)](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.spatial.procrustes.html#scipy.spatial.procrustes) and neutralization of analyze/digitization error effects (PC removal).
- **Elliptical Fourier Analysis (EFA):** Shape analysis of the entire contour using the [elliptical Fourier method](https://github.com/hbldh/pyefd).
- **Correlation Analysis:** Evaluates relationships between metrics and shape principal components (GPA PCA / EFA PCA).
- **Visualization:** Generates wing schematics, morphospaces (metric-to-shape regressions), consensus plots, and biplots for establishment.

---

## Installation and Dependencies

Requires Python 3.8+. Install the core dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install numpy pandas matplotlib scikit-learn scipy ipython
```

**Optional dependencies:**
- `pyefd` — for Elliptical Fourier Analysis (EFA): `pip install pyefd`
- `jinja2` — for colored correlation table output in Jupyter: `pip install jinja2`
- `openpyxl` — for reading `.xlsx` files: `pip install openpyxl`

---

## Quick Start (Workflow)

Below is a typical wing shape analysis workflow.

### 1. Initialization and Filtering
The class accepts a `pandas.DataFrame`. Data must contain point coordinates (columns `X1, Y1, X2, Y2...`) and any additional columns (sex, generation).

```python
import pandas as pd
from MorphoGet import morphoget

df = pd.read_csv("morphometric_data.csv")

# Initialization (set is_pre_aligned=True if data is already aligned)
get = gorphoget(df)

# Filtering (keep only males)
get.filter_data(column='sex', value='M')
```

### 2. Landmark Configuration
The `set_landmarks()` method tells the pipeline which columns are X and Y coordinates. Without arguments, it automatically finds `X<number>` and `Y<number>` columns.

```python
# Automatic detection
get.set_landmarks()

# Manual selection (e.g., points 14 through 30)
get.set_landmarks([f'{ax}{i}' for i in range(14, 31) for ax in ['X', 'Y']])
```

### 3. Procrustes Analysis (GPA) and Size Neutralization
Shape alignment and removal of the size component.

```python
get.run_gpa()

# Remove the first principal component
get.neutralize_pcs(components_to_remove=[0])

# Visualize the consensus shape
get.plot_consensus(flip_x=True, figsize=(5, 4), smooth=True)
```

### 4. Defining Key Points for Metric Calculation
Specify which point numbers are anatomically significant.

```python
get.set_fa_params(
    Basal=1, 
    Apex=5, 
    Costa=3, 
    Tornal=7, 
    TaperCosta=4,   # If BOTH Taper points are specified, taper metrics are automatically activated
    TaperMargin=6
)

# Verify the set points
get.print_fa_params()

# Visualize wing schematic with segments (L, C, Proximal, etc.)
get.plot_schematic(flip_x=True, figsize=(12, 7))
```

### 5. Calculating Flight Activity Metrics
Computes lengths, areas, and indices. The `comp` arguments specify which PCA component to use for finding inverse relationships (0 or 1).

```python
pipe.calculate_fa_metrics(
    ma_l_comp=1,     # Component for MA_l (Proximal_l, Distal_l)
    sa_l_comp=0,     # Component for SA_l (Costa_l, Tornal_l)
    sa_a_comp=1,     # Component for SA_a (Costa_A, Tornal_A)
    taper_l_comp=1   # Component for Taper_l (Taper, C)
)

# Visualize segment contributions to the component (vectors should point in opposite directions)
get.plot_biplot('ma_biplot', figsize=(4, 3))
```

### 6. Visualizing Shape Variation (Morphospace)
Allows visualization of how wing shape changes when a specific metric varies by ±Z standard deviations.

```python
get.plot_regressions(
    fa_metrics=['MA_s', 'AR_l', 'SA_a'], 
    n_cols=4,                            
    z_ext=3.0,                           # Deviation ±3σ
    flip_x=True,                         
    smooth=True,                         
    invert_sigma=['MA_s']                # Invert axis for visual comparison convenience
)
```

### 7. Principal Component Analysis of Shape (GPA PCA)
Visualizes shape deformations along PCs.

```python
get.plot_pca(
    pc_indices=[0, 1, 3], 
    n_cols=3, 
    z_ext=4.0, 
    smooth=True, 
    flip_x=True
)
```

### 8. Elliptical Fourier Analysis (EFA)
Shape analysis of the entire contour (requires `pyefd`).

```python
get.calculate_efa(order=15, normalize=True, use_neutralized=False)
get.run_efa_pca(n_components=5)

get.plot_efa_pca(pc_indices=[0, 1], n_cols=2, z_ext=4.0, flip_x=True)
```

### 9. Correlation Analysis
Evaluates relationships between metrics and shape components (requires `jinja2` for colored output).

```python
# Correlation of standard coordinates (Shape PCA) with metrics
get.show_correlations(fa_metrics=['MA_s', 'AR_l', 'SA_a'], n_pcs=4)

# Correlation of Elliptical Fourier components (EFA PCA) with metrics
get.show_efa_correlations(n_efa_pcs=4)
```

---

## Description of Calculated Metrics

Given appropriate wing geometry, the program computes key flight activity indicators using just 4 or 6 landmarks:

1. **Aspect Ratio** — characterizes wing elongation.
   - `AR_l`: Simple ratio of length to width (L/C).
   - `AR_s`: Ratio of squared wing length to wing area (L²/S).

2. **Second Moment of Area (Radius of Gyration)** — characterizes how close the widest part of the wing is to the body. Increased area farther from the body increases energy expenditure during flapping.
   - `MA_l`: Inverse relationship between proximal and distal dimensions (length segments L, divided by the chord).
   - `MA_s`: Ratio of areas divided by the chord (Proximal_s / (Proximal_s + Distal_s)). Useful for complex edge shapes.

3. **Sweep Angle** — the wing's angle of entry into air masses relative to the body's longitudinal axis. Related to lift generation.
   - `SA_l`: Inverse relationship between chord segment sizes divided by segment L.
   - `SA_a`: Inverse relationship of angles of the most recessed parts from the wing's longitudinal axis.

4. **Wing Taper** — calculated from two additional landmarks (TaperCosta, TaperMargin), characterizing wing narrowing toward the apex.
   - `Taper_l`: Inverse relationship between the taper segment and the chord.
   - `Taper_K`: Distal part shape index (from semicircle to triangle). Normalized so that a value close to 1 indicates a nearly perfect triangle. Calculated as the ratio of the actual triangle area to the reference area.

*Note:* In all cases, the inverse relationship is found as the second component of the ratio of segments/angles to each other. To verify the strength of the inverse relationship, use the `biplot` (contribution vectors should point in opposite directions).

---

## Visualization Parameters Reference

The `plot_consensus`, `plot_regressions`, `plot_pca`, and `plot_efa_pca` methods share these useful parameters:

| Parameter         | Description                                                                                                                                  
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------- 
| `figsize`              | Matplotlib figure size, e.g., `(5, 4)`                                                                                                   
| `flip_x`, `flip_y`   | Mirror the shape along the X and Y axes. Useful for left/right organs.                                                                    
| `smooth`             | `True` — smooth the contour with a cubic spline. `False` — connect points with straight lines.                                            
| `z_ext`                | Range of variation in standard deviations (σ). `z_ext=3.0` displays shape at -3σ, mean, and +3σ.                                           
| `invert_sigma`    | List of PC indices or metric names for which -σ and +σ should be swapped. Useful for logical interpretation of visualizations.           
| `n_cols`              | Number of columns in the plot grid.                                                                                                      

---

## License

- **MorphoGet:** [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) — open-source, modifications must be public.
- **Synthetic Data:** [CC0 1.0](http://creativecommons.org/publicdomain/zero/1.0/) — Public Domain, free for any use.

## References

- Kuhl, F. P., & Giardina, C. R. (1982). Elliptic Fourier features of a closed contour. *Computer Graphics and Image Processing*, 18(3), 236-258. [DOI: 10.1016/0146-664X(82)90034-X](https://doi.org/10.1016/0146-664X(82)90034-X)
- hbldh. (2016). PyEFD: Python implementation of "Elliptic Fourier Features of a Closed Contour". GitHub. [https://github.com/hbldh/pyefd](https://github.com/hbldh/pyefd)
```