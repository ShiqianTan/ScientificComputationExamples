import turtle
import math

def koch_line(t, length, depth):
    """
    Draw a Koch curve line segment.
    
    Args:
        t: turtle object
        length: length of the line segment
        depth: recursion depth (0 = straight line)
    """
    if depth == 0:
        t.forward(length)
    else:
        # Divide into 4 segments
        length = length / 3
        
        # First segment
        koch_line(t, length, depth - 1)
        
        # Turn left 60 degrees for the peak
        t.left(60)
        koch_line(t, length, depth - 1)
        
        # Turn right 120 degrees for the valley
        t.right(120)
        koch_line(t, length, depth - 1)
        
        # Turn left 60 degrees to continue
        t.left(60)
        koch_line(t, length, depth - 1)

def koch_snowflake(size, depth):
    """
    Draw a complete Koch snowflake.
    
    Args:
        size: side length of the initial triangle
        depth: recursion depth
    """
    # Set up the turtle
    screen = turtle.Screen()
    screen.bgcolor("white")
    screen.title(f"Koch Snowflake - Depth {depth}")
    screen.setup(800, 600)
    
    t = turtle.Turtle()
    t.speed(0)  # Fastest drawing
    t.color("blue")
    t.pensize(1)
    
    # Move to starting position
    t.penup()
    t.goto(-size/2, size/(2*math.sqrt(3)))
    t.pendown()
    
    # Draw three sides of the snowflake
    for i in range(3):
        koch_line(t, size, depth)
        t.right(120)  # Turn right 120 degrees for next side
    
    # Hide turtle and display
    t.hideturtle()
    screen.exitonclick()

def koch_snowflake_matplotlib(size, depth):
    """
    Alternative implementation using matplotlib for better visualization.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    def koch_points(start, end, depth):
        """Generate points for a Koch curve between two points."""
        if depth == 0:
            return [start, end]
        
        # Calculate the four key points
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # First third point
        p1 = [start[0] + dx/3, start[1] + dy/3]
        
        # Second third point
        p2 = [start[0] + 2*dx/3, start[1] + 2*dy/3]
        
        # Peak point (equilateral triangle)
        cx = (p1[0] + p2[0])/2
        cy = (p1[1] + p2[1])/2
        
        # Rotate 60 degrees
        angle = math.pi/3  # 60 degrees
        px = cx + (p1[0] - cx) * math.cos(angle) - (p1[1] - cy) * math.sin(angle)
        py = cy + (p1[0] - cx) * math.sin(angle) + (p1[1] - cy) * math.cos(angle)
        peak = [px, py]
        
        # Recursively generate points for each segment
        points = []
        segments = [(start, p1), (p1, peak), (peak, p2), (p2, end)]
        
        for seg_start, seg_end in segments:
            seg_points = koch_points(seg_start, seg_end, depth - 1)
            points.extend(seg_points[:-1])  # Avoid duplicate points
        
        points.append(end)
        return points
    
    # Create initial triangle vertices
    vertices = [
        [0, size/(math.sqrt(3))],
        [-size/2, -size/(2*math.sqrt(3))],
        [size/2, -size/(2*math.sqrt(3))],
        [0, size/(math.sqrt(3))]  # Close the triangle
    ]
    
    # Generate Koch curve points for each side
    all_points = []
    for i in range(3):
        side_points = koch_points(vertices[i], vertices[i+1], depth)
        all_points.extend(side_points[:-1])  # Avoid duplicate points
    
    # Close the shape
    all_points.append(all_points[0])
    
    # Extract x and y coordinates
    x_coords = [point[0] for point in all_points]
    y_coords = [point[1] for point in all_points]
    
    # Plot
    plt.figure(figsize=(10, 10))
    plt.plot(x_coords, y_coords, 'b-', linewidth=1)
    plt.fill(x_coords, y_coords, alpha=0.3, color='lightblue')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.title(f'Koch Snowflake - Depth {depth}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def animate_koch_snowflake(size, max_depth=5, interval=1000):
    """
    Create an animation of Koch snowflake evolution.
    
    Args:
        size: side length of the initial triangle
        max_depth: maximum recursion depth to animate
        interval: time between frames in milliseconds
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    
    def koch_points(start, end, depth):
        """Generate points for a Koch curve between two points."""
        if depth == 0:
            return [start, end]
        
        # Calculate the four key points
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # First third point
        p1 = [start[0] + dx/3, start[1] + dy/3]
        
        # Second third point
        p2 = [start[0] + 2*dx/3, start[1] + 2*dy/3]
        
        # Peak point (equilateral triangle)
        cx = (p1[0] + p2[0])/2
        cy = (p1[1] + p2[1])/2
        
        # Rotate 60 degrees
        angle = math.pi/3  # 60 degrees
        px = cx + (p1[0] - cx) * math.cos(angle) - (p1[1] - cy) * math.sin(angle)
        py = cy + (p1[0] - cx) * math.sin(angle) + (p1[1] - cy) * math.cos(angle)
        peak = [px, py]
        
        # Recursively generate points for each segment
        points = []
        segments = [(start, p1), (p1, peak), (peak, p2), (p2, end)]
        
        for seg_start, seg_end in segments:
            seg_points = koch_points(seg_start, seg_end, depth - 1)
            points.extend(seg_points[:-1])  # Avoid duplicate points
        
        points.append(end)
        return points
    
    def get_snowflake_data(depth):
        """Get x, y coordinates for snowflake at given depth."""
        # Create initial triangle vertices
        vertices = [
            [0, size/(math.sqrt(3))],
            [-size/2, -size/(2*math.sqrt(3))],
            [size/2, -size/(2*math.sqrt(3))],
            [0, size/(math.sqrt(3))]  # Close the triangle
        ]
        
        # Generate Koch curve points for each side
        all_points = []
        for i in range(3):
            side_points = koch_points(vertices[i], vertices[i+1], depth)
            all_points.extend(side_points[:-1])  # Avoid duplicate points
        
        # Close the shape
        all_points.append(all_points[0])
        
        # Extract x and y coordinates
        x_coords = [point[0] for point in all_points]
        y_coords = [point[1] for point in all_points]
        
        return x_coords, y_coords
    
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-size*0.6, size*0.6)
    ax.set_ylim(-size*0.4, size*0.4)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Initialize empty line and fill objects
    line, = ax.plot([], [], 'b-', linewidth=2)
    fill = ax.fill([], [], alpha=0.3, color='lightblue')[0]
    
    # Text for displaying current depth and properties
    text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame):
        depth = frame
        x_coords, y_coords = get_snowflake_data(depth)
        
        # Update line
        line.set_data(x_coords, y_coords)
        
        # Update fill
        fill.set_xy(list(zip(x_coords, y_coords)))
        
        # Calculate properties
        perim, area = snowflake_properties(size, depth)
        
        # Update text
        text.set_text(f'Depth: {depth}\nPerimeter: {perim:.1f}\nArea: {area:.1f}')
        
        # Update title
        ax.set_title(f'Koch Snowflake Evolution - Depth {depth}', fontsize=16)
        
        return line, fill, text
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=range(max_depth + 1),
                                 interval=interval, blit=False, repeat=True)
    
    plt.tight_layout()
    plt.show()
    
    return anim

# Example usage
if __name__ == "__main__":
    # Choose implementation
    use_animation = True  # Set to True for animation
    use_matplotlib = False  # Set to True for static plots
    use_turtle = False  # Set to True for turtle graphics
    
    if use_animation:
        print("Creating Koch snowflake animation...")
        print("The animation will show depths 0-5 with 1 second intervals")
        anim = animate_koch_snowflake(300, max_depth=5, interval=1000)
        # To save as GIF (optional, uncomment next line)
        # anim.save('koch_snowflake.gif', writer='pillow', fps=1)
    
    elif use_matplotlib:
        # Generate snowflakes with different depths
        for depth in range(0, 6):
            print(f"Generating Koch snowflake with depth {depth}")
            koch_snowflake_matplotlib(300, depth)
    
    elif use_turtle:
        # Use turtle graphics (interactive)
        depth = 3
        print(f"Drawing Koch snowflake with depth {depth}")
        print("Click on the window to close it")
        koch_snowflake(300, depth)
    
    # Calculate some properties
    def snowflake_properties(initial_length, depth):
        """Calculate perimeter and area of Koch snowflake."""
        # Perimeter grows by factor of 4/3 each iteration
        perimeter = 3 * initial_length * (4/3)**depth
        
        # Area approaches 8/5 of the original triangle's area
        original_area = (math.sqrt(3)/4) * initial_length**2
        area = original_area * (8/5) * (1 - (1/9) * (4/9)**depth) / (1 - 4/9)
        
        return perimeter, area
    
    # Show properties for different depths
    print("\nKoch Snowflake Properties:")
    print("Depth | Perimeter | Area")
    print("-" * 25)
    for d in range(6):
        perim, area = snowflake_properties(300, d)
        print(f"{d:5d} | {perim:8.1f} | {area:8.1f}")
