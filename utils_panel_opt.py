from ortools.sat.python import cp_model

def global_discrete_opt_quad_polygon(panel_edges, panel_labels, seam_edges):
    """
       Compute coordinates for polygonal panels ensuring minimal deformation and seam consistency.

       :param panel_edges: List of edge lengths per panel.
       :type panel_edges: list[list[float]]
       :param panel_labels: List of directional edge labels per panel. Directions: →(0), ↑(1), ←(2), ↓(3)
       :type panel_labels: list[list[int]]
       :param seam_edges: List of seam constraints: [panel1_index, (start1, end1), panel2_index, (start2, end2)]
       :type seam_edges: list[list[int, tuple, int, tuple]]
       :return:  x and y coordinates for each panel.
       :rtype: list[list[int]], list[list[int]]
       """
    model = cp_model.CpModel()
    num_panels = len(panel_edges)
    # Store decision variables for x and y coordinates of all panels
    panel_vars = []
    # Store objective terms for deformation minimization
    objective_terms = []

    for panel_idx in range(num_panels):
        turning_edge_lengths = panel_edges[panel_idx]
        turning_edge_labels = panel_labels[panel_idx]
        num_turning_points = len(turning_edge_lengths)
        num_points = len(turning_edge_lengths)

        # Define x and y coordinate decision variables for this panel
        x_vars = [model.NewIntVar(-500, 500, f'x_{panel_idx}_{i}') for i in range(num_turning_points)]
        y_vars = [model.NewIntVar(-500, 500, f'y_{panel_idx}_{i}') for i in range(num_turning_points)]
        panel_vars.append((x_vars, y_vars))

        # Minimize deviation from desired edge lengths
        for i in range(num_points):
            dx = model.NewIntVar(-500, 500, f'dx_{panel_idx}_{i}')
            dy = model.NewIntVar(-500, 500, f'dy_{panel_idx}_{i}')
            model.Add(dx == x_vars[(i + 1) % num_points] - x_vars[i])
            model.Add(dy == y_vars[(i + 1) % num_points] - y_vars[i])

            seg_length = model.NewIntVar(0, 100, f'length_{panel_idx}_{i}')
            model.AddAbsEquality(seg_length, dx + dy)

            deformation = model.NewIntVar(0, 100, f'deformation_{panel_idx}_{i}')#note: experimentally find longer edges are more stable, thus set the value to positive number
            model.AddAbsEquality(deformation, round(turning_edge_lengths[i]) - seg_length)

            deformation_squared = model.NewIntVar(0, 1000, f'deform_sq_{panel_idx}_{i}')
            model.AddMultiplicationEquality(deformation_squared, [deformation, deformation])
            objective_terms.append(deformation)

        # Directional constraints
        direction_bools = [
            [model.NewBoolVar(f'dir_{panel_idx}_{i}_{d}') for d in range(4)]
            for i in range(num_points)
        ]

        for i, label in enumerate(turning_edge_labels):
            for d in range(4):
                model.Add(direction_bools[i][d] == (1 if label == d else 0))

        # Add directional constraints
        for i in range(num_turning_points):
            next_i = (i + 1) % num_points

            # Right
            model.Add(x_vars[next_i] > x_vars[i]).OnlyEnforceIf(direction_bools[i][0])
            model.Add(y_vars[next_i] == y_vars[i]).OnlyEnforceIf(direction_bools[i][0])
            # Up
            model.Add(x_vars[next_i] == x_vars[i]).OnlyEnforceIf(direction_bools[i][1])
            model.Add(y_vars[next_i] > y_vars[i]).OnlyEnforceIf(direction_bools[i][1])
            # Left
            model.Add(x_vars[next_i] < x_vars[i]).OnlyEnforceIf(direction_bools[i][2])
            model.Add(y_vars[next_i] == y_vars[i]).OnlyEnforceIf(direction_bools[i][2])
            # Down
            model.Add(x_vars[next_i] == x_vars[i]).OnlyEnforceIf(direction_bools[i][3])
            model.Add(y_vars[next_i] < y_vars[i]).OnlyEnforceIf(direction_bools[i][3])


    # Seam constraints between panels
    for seam_idx, (p1, (start1, end1), p2, (start2, end2)) in enumerate(seam_edges):
        def segment_indices(start, end, n):
            return list(range(start, end)) if start < end else list(range(start, n)) + list(range(0, end))

        x1, y1 = panel_vars[p1]
        x2, y2 = panel_vars[p2]
        n1, n2 = len(x1), len(x2)

        indices1 = segment_indices(start1, end1, n1)
        indices2 = segment_indices(start2, end2, n2)

        def seam_length_vars(x_vars, y_vars, indices, tag):
            lengths = []
            for i in indices:
                ni = (i + 1) % len(x_vars)
                dx = model.NewIntVar(-100, 100, f'dx_{tag}_{i}')
                dy = model.NewIntVar(-100, 100, f'dy_{tag}_{i}')
                abs_dx = model.NewIntVar(0, 100, f'abs_dx_{tag}_{i}')
                abs_dy = model.NewIntVar(0, 100, f'abs_dy_{tag}_{i}')
                model.Add(dx == x_vars[i] - x_vars[ni])
                model.Add(dy == y_vars[i] - y_vars[ni])
                model.AddAbsEquality(abs_dx, dx)
                model.AddAbsEquality(abs_dy, dy)
                seg_len = model.NewIntVar(0, 100, f'seg_len_{tag}_{i}')
                model.Add(seg_len == abs_dx + abs_dy)
                lengths.append(seg_len)
            return lengths

        lengths1 = seam_length_vars(x1, y1, indices1, f'p1_{seam_idx}')
        lengths2 = seam_length_vars(x2, y2, indices2, f'p2_{seam_idx}')

        total1 = model.NewIntVar(0, 1000, f'total_len1_{seam_idx}')
        total2 = model.NewIntVar(0, 1000, f'total_len2_{seam_idx}')
        model.Add(total1 == sum(lengths1))
        model.Add(total2 == sum(lengths2))

        diff = model.NewIntVar(0, 1000, f'diff_{seam_idx}')
        model.AddAbsEquality(diff, total1 - total2)
        diff_squared = model.NewIntVar(0, 1000, f'diff_sq_{seam_idx}')
        model.AddMultiplicationEquality(diff_squared, [diff, diff])
        objective_terms.append(diff_squared)

    # Objective: minimize deformation and seam mismatch
    model.Minimize(sum(objective_terms))

    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        panel_coords_x, panel_coords_y = [], []
        for x_vars, y_vars in panel_vars:
            panel_coords_x.append([solver.Value(x) for x in x_vars])
            panel_coords_y.append([solver.Value(y) for y in y_vars])
        return panel_coords_x, panel_coords_y
    else:
        print('No optimal solution found.')
        return None, None

def get_panel_index(panel_name, panel_names):
    try:
        return panel_names.index(panel_name)
    except ValueError:
        raise ValueError(f"Panel '{panel_name}' not found in panel_names list")

def convert_stitches_to_lists(stitches, panel_names):
    # Convert list of stitches into a list of tuples using panel_names indices
    seam_lists = []  # To store the result
    for stitch_pair in stitches:
        panel1 = stitch_pair[0]['panel']
        edge1 = stitch_pair[0]['edge']
        panel2 = stitch_pair[1]['panel']
        edge2 = stitch_pair[1]['edge']

        # Get the indices of both panels based on their names in the panel_names list
        panel1_index = get_panel_index(panel1, panel_names)
        panel2_index = get_panel_index(panel2, panel_names)

        # Create a tuple with the format (panel1_index, edge1, panel2_index, edge2)
        seam_list = [panel1_index, edge1, panel2_index, edge2]
        seam_lists.append(seam_list)
    return seam_lists


