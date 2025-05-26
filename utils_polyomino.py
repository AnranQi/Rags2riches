import numpy as np

from holders import source_vertices, target_vertices
from svgpathtools import Path, Line, CubicBezier, Arc
from elements import Vertex

def sample_points(path):
    """
    Sample points at equal distances along a multi-segment path.

    Args:
        path: A path object supporting segment-wise iteration and global t-mapping.

    Returns:
        tuple:
            - List of sampled points (complex or coordinate format).
            - Corresponding global t values.
            - List of (segment_idx, point_type), where point_type ∈ {0=start, 'M'=middle}.
    """
    pts, t_vals, meta = [], [], []

    for seg_idx, seg in enumerate(path):
        seg_len = seg.length()
        n = int(np.floor(seg_len))

        # Add start point
        pts.append(seg.point(0))
        t_vals.append(path.t2T(seg, 0))
        meta.append((seg_idx, 0))

        for i in range(1, n):
            t = seg.ilength(i)
            p = seg.point(t)

            # Enforce min spacing of 0.1
            if pts and abs(p - pts[-1]) >= 0.1:
                pts.append(p)
                t_vals.append(path.t2T(seg, t))
                meta.append((seg_idx, 'M'))

    return pts, t_vals, meta




def angle_difference(n_vectors, reference_vector):
    """
    Compute the index of the vector (among n evenly spaced vectors on the unit circle)
    that has the smallest angular difference from a given reference vector.

    Parameters:
    - n_vectors (int): Number of unit vectors evenly spaced around the circle.
    - reference_vector (tuple or array): A 2D vector to compare against.

    Returns:
    - min_index (int): Index of the closest vector.
    - normalized_differences (np.ndarray): Array of normalized angle differences (0 to 1).
    """
    # Generate evenly spaced angles and corresponding unit vectors
    angles = np.linspace(0, 2 * np.pi, n_vectors, endpoint=False)
    vectors = np.stack((np.cos(angles), np.sin(angles)), axis=-1)

    # Compute angles of all vectors and the reference vector
    ref_angle = np.arctan2(reference_vector[1], reference_vector[0])
    vec_angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Compute absolute angular difference, normalized to [0, π]
    angle_diff = np.abs(vec_angles - ref_angle)
    angle_diff = np.where(angle_diff > np.pi, 2 * np.pi - angle_diff, angle_diff)

    # Normalize to [0, 1] by dividing by full circle (2π)
    normalized_differences = angle_diff / (2 * np.pi)

    # Return index of smallest difference and all normalized differences
    return np.argmin(normalized_differences), normalized_differences



def build_graph_cut(sampled_points):
    # Prepare edges for graph cut from sample points
    num_points = len(sampled_points)
    edges = []
    edge_weights = []

    # construct edges
    for i in range(1, num_points + 1):
        p1, p2 = sampled_points[(i - 1) % num_points], sampled_points[i % num_points]
        edge_weight = np.sqrt(np.power((p1.real - p2.real), 2) + np.power((p1.imag - p2.imag), 2))
        edges.append(((i - 1) % num_points, i % num_points))
        edge_weights.append(edge_weight)

    # Convert edge list to numpy array
    edges = np.array(edges)
    edge_weights = np.array(edge_weights)  # explain: this can not be int?

    # Calculate data cost based on angle differences
    data_cost = np.zeros((num_points, 4), dtype=np.int32)
    for i in range(1, num_points + 1):
        p1, p2 = sampled_points[(i - 1) % num_points], sampled_points[(i) % num_points]
        edge_vector = np.array([p2.real - p1.real, p2.imag - p1.imag])
        axis, angle_diff = angle_difference(4, edge_vector)
        angle_diff_0 = angle_diff[0]
        angle_diff_1 = angle_diff[1]
        angle_diff_2 = angle_diff[2]
        angle_diff_3 = angle_diff[3]

        data_cost[i - 1, 0] = int(angle_diff_0 * 1000)  # Scale to integers for pygco
        data_cost[i - 1, 1] = int(angle_diff_1 * 1000)  # Scale to integers for pygco
        data_cost[i - 1, 2] = int(angle_diff_2 * 1000)  # Scale to integers for pygco
        data_cost[i - 1, 3] = int(angle_diff_3 * 1000)  # Scale to integers for pygco

    # Create smooth cost
    smooth_cost = np.array([[0, 1, 2, 3], [1, 0, 1, 2], [2, 1, 0, 1], [3, 2, 1, 0]], dtype=np.int32)
    return edges, edge_weights, data_cost, smooth_cost




def compute_turning_point(labels, path_ts, pt_meta):
    """
    Identify turning points based on label changes and segment endpoints.

    Args:
        labels (list): Labels assigned to edges (e.g., from graph cut).
        path_ts (list): Global path 't' values for sampled points.
        pt_meta (list): List of (segment_id, point_type) where point_type ∈ {0=start, 1=end, 'M'=middle}.

    Returns:
        tuple:
            - List of turning point indices in the sample list.
            - Corresponding list of global 't' values for those turning points.
            - List of (start_idx, end_idx) tuples referring to each segment's turning points in sorted order.
    """
    tp_idx = []
    seg_starts = []

    # Detect turning points from label transitions
    for i in range(len(labels)):
        if labels[i] != labels[(i + 1) % len(labels)]:
            tp_idx.append((i + 1) % len(labels))

    # Include explicit segment start points
    for i, (_, t) in enumerate(pt_meta):
        if t == 0:
            if i not in tp_idx:
                tp_idx.append(i)
            seg_starts.append(i)

    # Sort turning point indices by path order
    tp_idx_sorted = sorted(tp_idx)
    tp_ts = [path_ts[i] for i in tp_idx_sorted]

    # Map segment start/end to sorted turning points
    seg_pairs = []
    for i, s in enumerate(seg_starts):
        s_idx = tp_idx_sorted.index(s)
        e_idx = tp_idx_sorted.index(seg_starts[(i + 1) % len(seg_starts)])
        seg_pairs.append((s_idx, e_idx))

    return tp_idx_sorted, tp_ts, seg_pairs




def compute_len_between_tps(path, tp_indices, tp_ts, point_labels):
    """
    Compute the length and directional label of each edge between consecutive turning points.

    Args:
        path: Path object with a `.length(t1, t2)` method.
        tp_indices (list): Indices of turning points in the sampled data.
        tp_ts (list): Global path time (t) values of turning points.
        point_labels (list): Labels assigned to sampled points (e.g., via graph cut).

    Returns:
        tuple:
            - List of edge lengths between turning points.
            - Corresponding list of directional labels.
    """
    edge_lens = []
    edge_labels = []

    # Fix spurious 0.0 t values not at the start
    for i in range(1, len(tp_ts)):
        if tp_ts[i] == 0.0:
            tp_ts[i] = 1.0

    for i in range(len(tp_ts)):
        if i < len(tp_ts) - 1:
            edge_lens.append(path.length(tp_ts[i], tp_ts[i + 1]))
        else:
            edge_lens.append(path.length() - sum(edge_lens))

        label = point_labels[tp_indices[i] % len(point_labels)]
        edge_labels.append(label)

    return edge_lens, edge_labels

def find_integer_points(panel, x_coords, y_coords, turning_point_path_t):
    """
    find the integer points along the paths formed by x_coords and y_coords, and build the vertices of source or target
    :param s_or_t:"source" or "target"
    :param x_coords:[40, 40, 25, 25, 0, 0, 40]
    :type x_coords: list of x coordinate in the shape
    :param y_coords: [0, -68, -68, -100, -100, 0, 0]
    :type y_coords:list of y coordinate in the shape
    :param turning_point_path_t: a list of t of points in the global path
    :param: global path() object
    :return: integer_points: list of points [(x0,y0), (x1, y1), ...]
    """
    s_or_t = panel.get_s_or_t()
    panel_name = panel.get_name()
    path = panel.get_path()

    # Store the integer points
    integer_points = []
    sample_points_on_curve = []  # the corresponding points on curve with integer_points

    # Number of vertices
    num_vertices = len(x_coords)
    boundary_vertices = []  # the list of boundary vertex classes

    # Iterate through each pair of points
    for i in range(num_vertices):
        line_points = []  # the points on one line(horizontal or vertical line)

        x1, y1 = x_coords[i], y_coords[i]
        x2, y2 = x_coords[(i + 1) % num_vertices], y_coords[(i + 1) % num_vertices]

        start_path_t = turning_point_path_t[i]
        end_path_t = turning_point_path_t[(i + 1) % num_vertices]

        if end_path_t < start_path_t:
            if end_path_t == 0:
                end_path_t = 1
            elif start_path_t == 1:
                start_path_t = 0
            else:
                print("check t's value")

        done_seg_length = path.length(0, start_path_t)  # the length of path that has been sampled

        quad_seg_length = 0  # the edge length
        # If the line segment is vertical
        if x1 == x2:
            # Vertical line: x is constant, vary y
            if y2 > y1:
                y_range = range(y1, y2)
            else:
                y_range = range(y1, y2, -1)
            for y in y_range:
                integer_points.append((x1, y))
                line_points.append((x1, y))
            quad_seg_length = max(y1, y2) - min(y1, y2)
        # If the line segment is horizontal
        elif y1 == y2:
            # Horizontal line: y is constant, vary x
            if x2 > x1:
                x_range = range(x1, x2)
            else:
                x_range = range(x1, x2, -1)
            for x in x_range:
                integer_points.append((x, y1))
                line_points.append((x, y1))
            quad_seg_length = max(x1, x2) - min(x1, x2)

        # todo: check if this depends on orientation of the curve
        seg_length = path.length(start_path_t, end_path_t)

        unit_sample_length = seg_length / quad_seg_length
        assert (len(line_points) == quad_seg_length)

        for m in range(len(line_points)):
            length = round(done_seg_length + m * unit_sample_length, 5)  # Round to 5 decimal places, otherwise, ilength() has numerial issues

            t = path.ilength(length)

            segment_index, segment_t = path.T2t(t)
            # explain: we are in counter clockwise setting
            if m == 0:
                if 1.0 - segment_t < 0.01:
                    segment_t = 0.0
                    segment_index = (segment_index + 1) % len(path)

            if s_or_t == "source":
                v = Vertex(len(source_vertices), s_or_t)
            elif s_or_t == "target":
                v = Vertex(len(target_vertices), s_or_t)
            else:
                print("not sure source or target, check")
            v.set_panel(panel_name)
            v.set_quad_x(line_points[m][0])
            v.set_quad_y(line_points[m][1])
            v.set_panel_path_number(len(path))

            v.set_global_t(t)
            v.set_segment_t(segment_t)
            v.set_segment_index(segment_index)
            v.set_type("boundary")  # explain: vertex only have "newcut" or "boundary" attributes

            curve_vertex = path.point(t)
            sample_points_on_curve.append(curve_vertex)
            v.set_curve_x(curve_vertex.real)
            v.set_curve_y(curve_vertex.imag)
            boundary_vertices.append(v)
            if s_or_t == "source":
                source_vertices.append(v)
            else:
                target_vertices.append(v)

    sample_points_on_curve_ = []  # change the points from complex representation to (x,y) format
    for p in sample_points_on_curve:
        sample_points_on_curve_.append((p.real, p.imag))
    return integer_points, sample_points_on_curve_, boundary_vertices


def shift_min_to_zero(lists):
    result = []
    for lst in lists:
        # Find the index of the minimum value
        min_value = min(lst)

        # Shift the list so that the minimum value is at index 0
        shifted_list = [v - min_value for v in lst]

        result.append(shifted_list)
    return result


# Function to reflect a path across the x-axis
def reflect_path_across_x_axis(path):
    reflected_path = Path()
    for segment in path:
        if isinstance(segment, Line):
            reflected_path.append(Line(
                start=segment.start.conjugate(),
                end=segment.end.conjugate()
            ))
        elif isinstance(segment, CubicBezier):
            reflected_path.append(CubicBezier(
                start=segment.start.conjugate(),
                control1=segment.control1.conjugate(),
                control2=segment.control2.conjugate(),
                end=segment.end.conjugate()
            ))
        elif isinstance(segment, Arc):
            # Reflect start and end points, and flip the sweep flag
            reflected_path.append(Arc(
                start=segment.start.conjugate(),
                radius=segment.radius,
                rotation=segment.rotation,
                large_arc=segment.large_arc,
                sweep=not segment.sweep,  # Flip the sweep flag
                end=segment.end.conjugate()
            ))
    return reflected_path
