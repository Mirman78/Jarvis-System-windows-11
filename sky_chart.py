import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

fig, ax = plt.subplots(figsize=(12, 10), facecolor='black')
ax.set_facecolor('black')
ax.set_title('🌌 Evening Sky - Oct 2026 (Sungai Petani)', fontsize=18, color='white')
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect('equal')
ax.axis('off')

# --- Stars ---
np.random.seed(42)
n_stars = 500
star_mags = np.clip(np.random.normal(4.0, 1.2, n_stars), 0, 6)
star_positions = np.random.rand(n_stars, 2) * 100
star_sizes = np.clip((6 - star_mags) ** 2 * 0.4, 0.3, 8)

ax.scatter(star_positions[:, 0], star_positions[:, 1],
           s=star_sizes, color='white', alpha=0.8, edgecolors='none')

# --- Horizon ---
ax.axhline(0, color='gray', linewidth=1, alpha=0.5)

# --- Sun (below horizon, so below y=0) ---
ax.scatter(50, -5, s=200, color='yellow', zorder=10,
           edgecolor='orange', linewidth=1, alpha=0.3)
ax.text(50, -10, 'Sun (below horizon)', fontsize=10, color='orange',
        ha='center', va='top')

# --- Moon ---
ax.scatter(75, 70, s=120, color='silver', zorder=9,
           edgecolor='white', linewidth=1)
ax.text(75, 75, 'Moon', fontsize=10, color='white', ha='center')

# --- Venus (low west = left side, low) ---
ax.scatter(10, 35, s=45, color='gold', zorder=8,
           edgecolor='white', linewidth=1)
ax.text(10, 30, 'Venus', fontsize=10, color='gold', ha='center')

# --- Jupiter (east = right side) ---
ax.scatter(85, 45, s=35, color='navajowhite', zorder=8,
           edgecolor='white', linewidth=1)
ax.text(85, 40, 'Jupiter', fontsize=10, color='navajowhite', ha='center')

# --- Mars (low west, near Venus) ---
ax.scatter(20, 20, s=18, color='red', zorder=7,
           edgecolor='darkred', linewidth=1)
ax.text(20, 15, 'Mars', fontsize=9, color='red', ha='center')

# --- Saturn (southeast = right side, low) ---
ax.scatter(60, 30, s=25, color='palegoldenrod', zorder=7,
           edgecolor='white', linewidth=1)
ax.add_patch(patches.Ellipse((60, 30), 6, 2, angle=30,
                              edgecolor='white', linewidth=0.5,
                              alpha=0.8, fill=False))
ax.text(60, 24, 'Saturn', fontsize=9, color='palegoldenrod', ha='center')

# --- Uranus / Neptune (high south) ---
ax.scatter(50, 80, s=10, color='lightblue', zorder=6,
           edgecolor='white', linewidth=0.5)
ax.text(50, 75, 'Uranus', fontsize=8, color='lightblue',
        ha='center', alpha=0.8)
ax.scatter(65, 85, s=9, color='royalblue', zorder=6,
           edgecolor='white', linewidth=0.5)
ax.text(65, 80, 'Neptune', fontsize=8, color='royalblue',
        ha='center', alpha=0.8)

# --- Notes: fixed positions below the chart, not random ---
notes = [
    "Venus: brightest planet, low in west after sunset",
    "Moon: high in south-east",
    "Jupiter: rises after midnight, bright in east",
    "Saturn: best seen before dawn, southeast",
    "Mars: reddish, low in west early evening",
    "Uranus / Neptune: need binoculars, high in south",
    "Best viewing: clear cold breeze = excellent visibility",
]
for i, note in enumerate(notes):
    fig.text(0.02, 0.18 - i * 0.028, note, fontsize=9, color='white',
             ha='left', va='top',
             bbox=dict(facecolor='black', alpha=0.6, edgecolor='none'))

# --- Colorbar bound to real data (mag 0 = bright = dark on the ramp) ---
sm = ScalarMappable(norm=Normalize(vmin=0, vmax=6), cmap=plt.cm.gray_r)
cbar = fig.colorbar(sm, ax=ax, orientation='horizontal',
                    fraction=0.03, pad=0.02)
cbar.set_label('Magnitude (brighter = lower number)', color='white')
cbar.ax.xaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax, 'xticklabels'), color='white')

plt.tight_layout()
plt.savefig('sky_chart_oct2026.png', dpi=150, facecolor='black',
            bbox_inches='tight')
print("Sky chart saved as 'sky_chart_oct2026.png'")