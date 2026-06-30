README - Synthetic Dataset "Noisy Octagons"

1. General Description

This dataset contains n = 1000 artificially generated octagon instances.

Each object is represented by the coordinates of eight vertices (x, y), with additive Gaussian noise applied to each coordinate.

The ideal underlying shape is a horizontally elongated octagon:

The leftmost and rightmost extreme points are located at x = ±3.
The uppermost and lowermost extreme points are located at y = ±1.5.

The added noise simulates natural shape variability, analogous to the outline variation observed in insect wing plates.

2. Generation Parameters

Total number of shapes: n = 1000 (user-defined; any value may be specified)
Number of vertices: 8 (closed polygon)

Ideal Geometry

The shape is centered at the origin (0, 0).

Vertex angles (in degrees, measured from the positive X-axis):

180, 135, 90, 45, 0, -45, -90, -135

This corresponds to a counterclockwise traversal starting from the leftmost point.

Ideal vertex coordinates:

x = 3 cos(theta)
y = 1.5 sin(theta)

where theta is expressed in radians.

Added Noise

Independent Gaussian noise is applied to each coordinate.

Mean: mu = 0
Standard deviation: sigma = 0.1

Noise is generated independently for both x and y coordinates of every vertex.

3. Data File Structure

The dataset is stored as a tabular DataFrame with 16 columns:

x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6, x7, y7, x8, y8

Each row corresponds to one octagonal shape.

Vertex Ordering

Vertices follow the angular sequence listed above, beginning with:

(x1, y1) = angle 180 (leftmost point)

and proceeding counterclockwise.

The generating code closes the polygon by connecting vertex 8 back to vertex 1.

This closure is not explicitly stored in the dataset, so users should close the contour manually during visualization or analysis.

4. Example Usage

import pandas as pd
import matplotlib.pyplot as plt

from MorphometricClass import MorphometricPipeline
from SilicData import df

5. Visualization

Plotting routines are included in the generation script.

For quick visual inspection, the script displays the first five generated shapes.

Example outputs illustrate noisy variations around the ideal elongated octagonal structure.

6. Authorship and Licensing

This synthetic dataset was created for research and educational purposes.

Generated using Python, NumPy, Pandas, and Matplotlib.

Generation date: see file metadata

License: Creative Commons Attribution 4.0 International License.

7. Notes

The random seed is not fixed, so rerunning the generator will produce different noise realizations.

At sigma = 0.1, the octagonal structure remains clearly recognizable.

Increasing sigma produces progressively stronger distortions and may eventually destroy the geometric structure.

The ideal octagon is not regular. It is stretched along the X-axis with a semi-axis ratio of 2:1.
