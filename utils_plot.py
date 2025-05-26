import numpy as np
import matplotlib.pyplot as plt


def plot_quad_panel_dot(ax: plt.Axes, vertices: np.ndarray, color: str, s: int = 5, title: str = "None",
                        label: str = None) -> None:
    """
    Plot a quad panel using vertices as individual dots.

    :param ax: Matplotlib axis to plot on.
    :param vertices: A (4, 2) array of vertices for the quad.
    :param color: Color of the points.
    :param s: Marker size.
    :param title: Text annotation near the first vertex.
    :param label: Label for the legend.
    """
    # Close the loop by adding the first vertex at the end
    closed_vertices = np.vstack([vertices, vertices[0]])

    ax.plot(closed_vertices[:, 0], closed_vertices[:, 1], '.', markersize=s, color=color, label=label)
    ax.text(np.mean(closed_vertices[:, 0]), np.mean(closed_vertices[:, 1]), title, color=color)


def plot_quad_panel_line(ax: plt.Axes, edges: np.ndarray, color: str, s: int = 5, annotation: str = "None",
                         label: str = None) -> None:
    """
    Plot a quad panel as lines using edge pairs.

    :param ax: Matplotlib axis to plot on.
    :param edges: An array of shape (8, 2) representing four edges with start and end vertices.
    :param color: Color of the lines.
    :param s: Marker size.
    :param annotation: Annotation text at the center.
    :param label: Label for the legend (added only on first edge).
    """
    num_edges = len(edges) // 2
    for i in range(num_edges):
        start, end = edges[2 * i], edges[2 * i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], '.-', markersize=s, color=color,
                label=label if i == 0 else None)

    center = np.mean(edges, axis=0)
    ax.text(center[0], center[1], annotation, color=color)


def plot_curve_panel_line(ax: plt.Axes, edges, translation_vector, color, s=5, title="None", label=None):
    """
    Plot a curve panel with translation applied.

    :param ax: Matplotlib axis to plot on.
    :param edges: Sequence of edge start/end vertices [start, end, start, end, ...]
    :param translation_vector: 2D translation vector
    :param color: Line color
    :param s: Marker size
    :param title: Annotation text
    :param label: Optional label for legend
    """
    # Ensure proper array shape
    edges = np.array(edges).reshape(-1, 2)
    translation_vector = np.array(translation_vector)

    # Apply translation to all vertices
    translated_edges = edges + translation_vector

    # Plot the edges
    num_edges = len(translated_edges) // 2
    for i in range(num_edges):
        start = translated_edges[2 * i]
        end = translated_edges[2 * i + 1]
        ax.plot([start[0], end[0]], [start[1], end[1]], '.-', markersize=s, color=color, label=label if i == 0 else None)

    # Annotate the center of the translated shape
    center = np.mean(translated_edges, axis=0)
    ax.text(center[0], center[1], title, color=color)



def plot_curve_panel_line_(ax, vertices, translation_vector, color, s=5, title="None",label=None):
    """
  # explain: plot the original curve panel by edges
    :param ax:
    :type ax:
    :param vertices: vertices is a format of [edge_start_vertex, edge_end_vertex, edge_start_vertex, edge_end_vertex, ...]
    :type vertices:
    :param translation_vector:  #  translate curve center to align with the quad panel center
    :type translation_vector:
    :param color:
    :type color:
    :param s:
    :type s:
    :param title:
    :type title:
    :param label:
    :type label:
    """

    # Close the loop by appending the first point at the end
    vertices = np.append(vertices, [vertices[0]], axis=0)

    # Plot the lines connecting the points
    translation_vector_x = translation_vector[0]
    translation_vector_y = translation_vector[1]

    for i in range(int(len(vertices)/2)):
        ax.plot([vertices[2*i, 0]+translation_vector_x, vertices[2*i+1, 0]+translation_vector_x],
                [vertices[2*i, 1]+translation_vector_y, vertices[2*i+1, 1]+translation_vector_y], '.-', markersize=s, color=color)
    ax.text(np.mean(vertices[:, 0])+translation_vector_x, np.mean(vertices[:, 1]) +translation_vector_y, title,color=color)