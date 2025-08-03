import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with correct physical dimensions
fig_width_in = 6  # 6 inches wide
fig_height_in = 2  # 2 inches tall
dpi = 100  # Standard screen resolution

# Create figure with physical inch dimensions
fig, ax1 = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi), plt.gca()

# Create x-axis in inches (0 to 5 inches)
x_inches = np.linspace(0, 5, 100)
ax1.plot(x_inches, np.sin(x_inches), 'b-')
ax1.set_xlabel('Inches')
ax1.set_ylabel('Sin(inches)', color='b')
ax1.tick_params('y', colors='b')

# Create second axis with centimeters
ax2 = ax1.twiny()  # Create a twin axis on top
cm_per_inch = 2.54  # Conversion factor
ax2.set_xlim(ax1.get_xlim()[0] * cm_per_inch, ax1.get_xlim()[1] * cm_per_inch)
ax2.set_xlabel('Centimeters')
ax2.spines['top'].set_position(('axes', 1.1))  # Move top axis up slightly

# Add physical scale markers
for x in range(6):  # Every inch
    ax1.axvline(x, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x * cm_per_inch, color='gray', linestyle=':', alpha=0.5)

# Add minor ticks to show precise measurements
ax1.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
ax2.xaxis.set_minor_locator(plt.MultipleLocator(0.1 * cm_per_inch))

plt.title('Dual Axis: Inches (bottom) and Centimeters (top)')
plt.tight_layout()

# Save with high DPI for printing
plt.savefig('dual_axis_physical.png', dpi=300, bbox_inches='tight')
plt.show()
