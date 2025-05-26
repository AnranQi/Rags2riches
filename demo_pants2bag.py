import time
import sys
import gco
import matplotlib.pyplot as plt
import numpy as np
from ortools.sat.python import cp_model
from skimage.util.shape import view_as_windows

# Internal modules
from elements import Panel, Cycle, Pattern, set_seam_pair_index
from pattern import wrappers
from utils_quads import *
from utils_panel_opt import *
from utils_stitches import *
from utils_plot import *
from utils_polyomino import *
from utils_swindows import *

source_design_tool = 'neuraltailor'  # Options: 'adobeai', 'neuraltailor',
source_pant_pattern_file = f'./data/pants_neuraltailor/specification.json'
source_pattern = wrappers.VisPattern(source_pant_pattern_file)

# Target pattern setup (drawn in Illustrator)
target_design_tool = 'adobeai'
target_panel_dir = f'./data/target_design'
target_panel_names = ['rect1', 'rect2', 'rect3', 'rect4', 'square', 'handle1', 'handle2']
target_pattern = Pattern(target_panel_dir, target_panel_names)

source_panels = []
target_panels = []

# Unit scaling factors
default_scale = 1 / 28.3465  # Convert pt (from SVG) to cm
source_scale = 0.25
target_scale = default_scale * 0.25

# Weight parameters for optimization terms
fabrication_weight = -75
fabrication_reused_weight = -fabrication_weight
dg_seam_reused_weight = -500
dg_newcut_reused_weight = -500
dg_boundary_reused_weight = -500
iou_weight = 100
iou_threshold = 1.0

# Reuse bonus incentives
newcut_reuse_as_seam_bonus = 5
seam_reuse_bonus = 10

# Process both source and target patterns
print('start processing patterns')
for pattern_index, pattern in enumerate([source_pattern, target_pattern]):
    panels = []
    panels_tp_tss = []
    panels_turning_edge_lengths = []
    panels_turning_edge_labels = []

    s_or_t = "source" if pattern_index == 0 else "target"
    panel_names = list(pattern.pattern['panels'].keys())
    stitches = pattern.get_stitch()
    seam_lists = convert_stitches_to_lists(stitches, panel_names)

    for panel_index, panel_name in enumerate(panel_names):
        path = pattern._read_a_panel(panel_name)

        if (s_or_t == 'target' and target_design_tool == 'adobeai') or \
                (s_or_t == 'source' and source_design_tool == 'adobeai'):
            path = reflect_path_across_x_axis(path)

        path, reversed_flag = pattern.ensure_counterclockwise(panel_name, path)
        seam_lists = update_seams_if_reversed(seam_lists, panel_index, path, reversed_flag)

        # Apply scale factor
        scale = source_scale if s_or_t == 'source' else target_scale
        path = path.scaled(scale)

        # Construct Panel object
        panel = Panel(panel_index, path, panel_name, s_or_t, pattern)
        panels.append(panel)
        (source_panels if s_or_t == 'source' else target_panels).append(panel)

        # Sample boundary points
        sampled_points, sampled_ts, segment_indicators = sample_points(path)

        # Prepare data for graph cut
        edges_array, edge_weights, data_cost, smooth_cost = build_graph_cut(sampled_points)
        labels = gco.cut_general_graph(edges_array, edge_weights, data_cost, smooth_cost, n_iter=1)

        # Compute turning points
        tp_indices, tp_ts, seg_start_end = compute_turning_point(labels, sampled_ts, segment_indicators)
        panels_tp_tss.append(tp_ts)

        edge_lengths, edge_labels = compute_len_between_tps(path, tp_indices, tp_ts, labels)
        panels_turning_edge_lengths.append(edge_lengths)
        panels_turning_edge_labels.append(edge_labels)

        # Update seam index structure based on turning point indices
        seam_lists = replace_edges_with_tp(seam_lists, panel_index, seg_start_end)

    # Solve discrete ILP to get vertex positions of quad polygons
    quad_x_coords, quad_y_coords = global_discrete_opt_quad_polygon(
        panels_turning_edge_lengths,
        panels_turning_edge_labels,
        seam_lists
    )

    quad_x_coords = shift_min_to_zero(quad_x_coords)
    quad_y_coords = shift_min_to_zero(quad_y_coords)

    # Map curve points to integer grid and build quads for each panel
    for idx, panel in enumerate(panels):
        quad_x = quad_x_coords[idx]
        quad_y = quad_y_coords[idx]
        tp_ts = panels_tp_tss[idx]

        integer_pts, curve_pts, integer_vertices = find_integer_points(panel, quad_x, quad_y, tp_ts)
        panel.set_boundary_integer_points(integer_pts)
        panel.set_boundary_curve_points(curve_pts)

        # Compute curve-to-grid translation
        tx = np.mean([pt[0] for pt in integer_pts]) - np.mean([pt[0] for pt in curve_pts])
        ty = np.mean([pt[1] for pt in integer_pts]) - np.mean([pt[1] for pt in curve_pts])
        panel.set_translation_curve_to_quad_panel([tx, ty])

        # Build quad mesh and compute gradients
        faces, verts, verts_def = build_quads_from_polygon(panel)
        panel.set_seams()
        edge_indices = compute_edges(faces, verts)
        centers, grads, b_centers, b_grads = compute_deformation_gradients_edge(panel, verts, verts_def, edge_indices)

# Assign seam pair indices for each pattern
set_seam_pair_index(source_edges, source_pattern)
set_seam_pair_index(target_edges, target_pattern)

# ----------------------- Cycle Extraction -----------------------

source_cycles = []  # List to store all source cycle objects
t0 = time.time()

# Perform sliding window to extract all valid source cycles that can align with target panels
for sp in source_panels:
    sp_index = sp.get_index()
    sp_name = sp.get_name()
    sp_bbox = sp.get_bounding_box()
    sp_quad_x_min, sp_quad_y_min, sp_quad_x_max, sp_quad_y_max = sp_bbox
    sp_index_mask_shift1 = sp.get_panel_index_mask_shift1()

    for tp in target_panels:
        tp_index = tp.get_index()
        tp_name = tp.get_name()
        tp_num_quads = tp.get_quads_number()
        tp_index_mask_shift1 = tp.get_panel_index_mask_shift1()
        tp_bbox = tp.get_bounding_box()
        tp_quad_x_min, tp_quad_y_min, tp_quad_x_max, tp_quad_y_max = tp_bbox

        # Compute default translation between source and target panel origins
        default_translation = (
            sp_quad_x_min - tp_quad_x_min,
            sp_quad_y_min - tp_quad_y_min
        )

        # Skip if source panel is smaller than target panel
        if sp_index_mask_shift1.shape[0] < tp_index_mask_shift1.shape[0] or \
                sp_index_mask_shift1.shape[1] < tp_index_mask_shift1.shape[1]:
            continue

        # Use sliding window to extract patches from source matching target's shape
        possible_cycles = view_as_windows(
            sp_index_mask_shift1,
            window_shape=tp_index_mask_shift1.shape
        )
        assert possible_cycles[0, 0].shape == tp_index_mask_shift1.shape

        for x in range(possible_cycles.shape[0]):
            for y in range(possible_cycles.shape[1]):
                window = possible_cycles[x, y]
                polyomino = (window != 0) & (tp_index_mask_shift1 != 0)

                # Extract (source_quad_idx, target_quad_idx) pairs where both are valid
                st_pairs = [
                    (int(window[i, j] - 1), int(tp_index_mask_shift1[i, j] - 1))
                    for i in range(window.shape[0])
                    for j in range(window.shape[1])
                    if polyomino[i, j]
                ]

                # Compute IoU over quads
                iou = len(st_pairs) / tp_num_quads
                if iou >= iou_threshold:
                    cycle_id = len(source_cycles)
                    cycle = Cycle(cycle_id)
                    edge_pairs = []

                    for s_idx, t_idx in st_pairs:
                        for s_edge, t_edge in zip(source_quads[s_idx].get_edges(), target_quads[t_idx].get_edges()):
                            edge_pairs.append((s_edge, t_edge))
                            s_edge.set_cycle_indices(cycle_id)
                            t_edge.set_cycle_indices(cycle_id)

                    # Set cycle attributes
                    cycle.set_polyomino(polyomino)
                    cycle.set_translation([x, y])
                    cycle.set_source_panel_index(sp_index)
                    cycle.set_source_panel_name(sp_name)
                    cycle.set_target_panel_index(tp_index)
                    cycle.set_target_panel_name(tp_name)
                    cycle.set_st_edge_pairs(edge_pairs)
                    cycle.set_bounding_box(
                        tp_quad_x_min + default_translation[0] + x,
                        tp_quad_y_min + default_translation[1] + y,
                        tp_quad_x_max - tp_quad_x_min,
                        tp_quad_y_max - tp_quad_y_min
                    )
                    cycle.set_quad_iou(iou)
                    source_cycles.append(cycle)

print('finish processing patterns')
print(f"number of cycles {len(source_cycles)}")

# ----------------------- Constraint Model Setup -----------------------
model = cp_model.CpModel()

num_sc = len(source_cycles)
num_sp = len(source_panels)
num_tc = len(target_panels)

# Create selection variables p[i] ∈ {0,1}
p = [model.NewBoolVar(f'p[{i}]') for i in range(num_sc)]

# Create reuse variables reuse[i][j] ∈ {0,1} for i < j
reuse = [
    [model.NewBoolVar(f'reuse[{i},{j - (i + 1)}]') for j in range(i + 1, num_sc)]
    for i in range(num_sc)
]

# Add logical AND reuse constraints: reuse[i][j] = p[i] ∧ p[j]
for i in range(num_sc):
    for j in range(i + 1, num_sc):
        r = reuse[i][j - (i + 1)]
        model.Add(r <= p[i])
        model.Add(r <= p[j])
        model.Add(r >= p[i] + p[j] - 1)

# ----------------------- Reuse Weights Initialization -----------------------
fabrication_reused_values = [np.zeros(num_sc - (i + 1)) for i in range(num_sc)]
dg_seam_reused_values = [np.zeros(num_sc - (i + 1)) for i in range(num_sc)]
dg_boundary_reused_values = [np.zeros(num_sc - (i + 1)) for i in range(num_sc)]
dg_newcut_reused_values = [np.zeros(num_sc - (i + 1)) for i in range(num_sc)]

# ----------------------- Reuse Value Computation & Conflict Constraints -----------------------
for i in range(len(source_cycles)):
    for j in range(i + 1, len(source_cycles)):
        # === Fabrication Reuse Metrics ===
        seam_reuse_score = 0  # amount of seam reuse (source and target are both seams)
        newcut_as_seam_score = 0  # reuse of newcut edges from source as seams in target
        total_fabrication_score = 0  # general reuse count, aggregated

        # === Deformation Gradient Metrics ===
        seam_dg_source1 = 0  # DG for seam reuse in cycle1
        seam_dg_source2 = 0  # DG for seam reuse in cycle2
        newcut_dg_source1 = 0  # DG for newcut reuse in cycle1
        newcut_dg_source2 = 0  # DG for newcut reuse in cycle2
        boundary_dg_source1 = 0  # DG for boundary reuse in cycle1
        boundary_dg_source2 = 0  # DG for boundary reuse in cycle2

        # Extract cycles and panel information
        cycle1 = source_cycles[i]
        cycle2 = source_cycles[j]

        index1 = cycle1.get_index()
        index2 = cycle2.get_index()

        source_panel_1 = cycle1.get_source_panel_name()
        source_panel_2 = cycle2.get_source_panel_name()
        target_panel_1 = cycle1.get_target_panel_name()
        target_panel_2 = cycle2.get_target_panel_name()

        num_edges_target1 = target_panels[cycle1.get_target_panel_index()].get_number_of_border_edges()
        num_edges_target2 = target_panels[cycle2.get_target_panel_index()].get_number_of_border_edges()

        # === Case 1: Boundary Edge Reuse ===
        for (s_edge, t_edge) in cycle1.get_st_edge_pairs():
            if (s_edge.get_type() == 'boundary' and t_edge.get_type() == 'boundary') or \
                    (s_edge.get_type() == 'seam' and t_edge.get_type() == 'boundary') or \
                    (s_edge.get_type() == 'boundary' and t_edge.get_type() == 'seam'):
                boundary_dg_source1 += np.linalg.norm(
                    s_edge.get_deformation_gradient() - t_edge.get_deformation_gradient(), 'fro')
        dg_boundary_reused_values[i][j - (i + 1)] += boundary_dg_source1 / num_edges_target1

        for (s_edge, t_edge) in cycle2.get_st_edge_pairs():
            if (s_edge.get_type() == 'boundary' and t_edge.get_type() == 'boundary') or \
                    (s_edge.get_type() == 'seam' and t_edge.get_type() == 'boundary') or \
                    (s_edge.get_type() == 'boundary' and t_edge.get_type() == 'seam'):
                boundary_dg_source2 += np.linalg.norm(
                    s_edge.get_deformation_gradient() - t_edge.get_deformation_gradient(), 'fro')
        dg_boundary_reused_values[i][j - (i + 1)] += boundary_dg_source2 / num_edges_target2

        # === Case 2: Same Source Panel → Check for Overlap ===
        if source_panel_1 == source_panel_2:
            bboxes1 = [
                [box[0] + cycle1.get_translation()[0], box[1] + cycle1.get_translation()[1], box[2], box[3]]
                for box in cycle1.get_divided_bounding_boxes()
            ]
            bboxes2 = [
                [box[0] + cycle2.get_translation()[0], box[1] + cycle2.get_translation()[1], box[2], box[3]]
                for box in cycle2.get_divided_bounding_boxes()
            ]
            if check_intersections(bboxes1, bboxes2):
                model.Add(reuse[i][j - (i + 1)] == 0)

            # === Case 2b: Reuse Newcut as Seam Between Target Panels ===
            if target_panel_2 in target_pattern._get_globle_stitch_panel(target_panel_1):
                source_newcuts1, target_newcuts1 = cycle1.get_st_edges_based_tseam()
                source_newcuts2, target_newcuts2 = cycle2.get_st_edges_based_tseam()

                for a in range(len(source_newcuts1)):
                    for b in range(len(source_newcuts2)):
                        if source_newcuts1[a].get_pair_index() == source_newcuts2[b].get_index() and \
                                target_newcuts1[a].get_pair_index() == target_newcuts2[b].get_index():
                            newcut_as_seam_score += newcut_reuse_as_seam_bonus
                            newcut_dg_source1 += np.linalg.norm(
                                source_newcuts1[a].get_deformation_gradient() - target_newcuts1[
                                    a].get_deformation_gradient(), 'fro')
                            newcut_dg_source2 += np.linalg.norm(
                                source_newcuts2[b].get_deformation_gradient() - target_newcuts2[
                                    b].get_deformation_gradient(), 'fro')
                            break
            fabrication_reused_values[i][j - (i + 1)] = (newcut_as_seam_score / num_edges_target1) + (
                        newcut_as_seam_score / num_edges_target2)
            dg_newcut_reused_values[i][j - (i + 1)] = (newcut_dg_source1 / num_edges_target1) + (
                        newcut_dg_source2 / num_edges_target2)

        # === Case 3: Seam Reuse Between Targets and Sources ===
        if target_panel_2 in target_pattern._get_globle_stitch_panel(target_panel_1) and \
                source_panel_2 in source_pattern._get_globle_stitch_panel(source_panel_1):

            source_seams1, target_seams1 = cycle1.get_st_edges_based_stseam()
            source_seams2, target_seams2 = cycle2.get_st_edges_based_stseam()

            for a in range(len(source_seams1)):
                for b in range(len(source_seams2)):
                    if source_seams1[a].get_pair_index() == source_seams2[b].get_index() and \
                            target_seams1[a].get_pair_index() == target_seams2[b].get_index():
                        seam_reuse_score += seam_reuse_bonus
                        seam_dg_source1 += np.linalg.norm(
                            source_seams1[a].get_deformation_gradient() - target_seams1[a].get_deformation_gradient(),
                            'fro')
                        seam_dg_source2 += np.linalg.norm(
                            source_seams2[b].get_deformation_gradient() - target_seams2[b].get_deformation_gradient(),
                            'fro')
                        break

            fabrication_reused_values[i][j - (i + 1)] += (seam_reuse_score / num_edges_target1) + (
                        seam_reuse_score / num_edges_target2)
            dg_seam_reused_values[i][j - (i + 1)] = (seam_dg_source1 / num_edges_target1) + (
                        seam_dg_source2 / num_edges_target2)

# === Target Constraints: Each target panel should receive exactly one source cycle ===
for i in range(num_tc):
    relevant_source_cycles = [p[j] for j in range(num_sc) if source_cycles[j].get_target_panel_index() == i]
    model.add(sum(relevant_source_cycles) == 1)
# === Compute Cost Matrix Based on Edge Pairings ===
cost_matrix = []
for s_cycle in source_cycles:
    edge_pairs = s_cycle.get_st_edge_pairs()
    num_target_edges = target_panels[s_cycle.get_target_panel_index()].get_number_of_border_edges()
    cost = 0
    for s_edge, t_edge in edge_pairs:
        s_type = s_edge.get_type()
        t_type = t_edge.get_type()
        if s_type == "seam" and t_type == "seam":
            cost += 4
        elif s_type == "seam" and t_type == "boundary":
            cost += 3
        elif s_type == "newcut" and t_type == "seam":
            cost += 2
        elif s_type == "newcut" and t_type == "boundary":
            cost += 1
        elif s_type == "boundary" and t_type == "seam":
            cost += 2
        # boundary-boundary has no added cost
    cost_matrix.append(cost / num_target_edges)

# === Objective Function Construction ===
reuse_flat = [item for sublist in reuse for item in sublist]
fabrication_flat = [item for sublist in fabrication_reused_values for item in sublist]
dg_seam_flat = [item for sublist in dg_seam_reused_values for item in sublist]
dg_newcut_flat = [item for sublist in dg_newcut_reused_values for item in sublist]
dg_boundary_flat = [item for sublist in dg_boundary_reused_values for item in sublist]

assert len(reuse_flat) == len(fabrication_flat)

obs = []
for i in range(len(reuse_flat)):
    reuse_score = (int(fabrication_flat[i] * fabrication_reused_weight) +
                   int(dg_seam_flat[i] * dg_seam_reused_weight) +
                   int(dg_newcut_flat[i] * dg_newcut_reused_weight) +
                   int(dg_boundary_flat[i] * dg_boundary_reused_weight))
    obs.append(reuse_flat[i] * reuse_score)

# Add cost and IOU to the objective
for i in range(num_sc):
    cost_term = int(fabrication_weight * cost_matrix[i]) * p[i]
    iou_term = int(source_cycles[i].get_quad_iou() * iou_weight) * p[i]
    obs.append(cost_term + iou_term)

# === Solve the Optimization Problem ===
print('start sovling')
model.maximize(sum(obs))
t0 = time.time()
solver = cp_model.CpSolver()
solver.parameters.num_search_workers = 12
status = solver.Solve(model)

if status == cp_model.OPTIMAL:
    print(f'Total cost = {solver.ObjectiveValue()}')
    print(f"Solving time: {time.time() - t0:.2f}s")
else:
    print('No solution found.')
    sys.exit(1)

# === Visualization ===
print("Plotting result (no deformation applied)")
assigned_cycles = [source_cycles[i] for i in range(num_sc) if solver.BooleanValue(p[i])]

fig_curve, ax_curve = plt.subplots()
fig_quad, ax_quad = plt.subplots()
fix_margin = 30

# Plot source panels
for sp in source_panels:
    margin = sp.get_index() * fix_margin
    boundary = [[x + margin, y] for x, y in sp.get_boundary_integer_points()]
    plot_quad_panel_dot(ax_quad, boundary, color='blue', title=sp.get_name(), label='Source Panel')

    vertices = []
    translation = sp.get_translation_curve_to_quad_panel()
    for quad in sp.get_quads():
        for edge in quad.get_edges():
            if edge.get_type() in ("seam", "boundary"):
                vertices.append([edge.get_start_vertex_x() + margin, edge.get_start_vertex_y()])
                vertices.append([edge.get_end_vertex_x() + margin, edge.get_end_vertex_y()])
    plot_curve_panel_line(ax_curve, vertices, translation, color='black', s=5,
                          title=sp.get_name(), label='Source Curve Panel')

# Plot matched target panels
for cycle in assigned_cycles:
    tp = target_panels[cycle.get_target_panel_index()]
    margin = cycle.get_source_panel_index() * fix_margin

    # Align centers
    tp_cx, tp_cy = tp.get_panel_quad_bounding_box_center()
    sc_cx, sc_cy = cycle.get_bounding_box_center()
    translation = [(sc_cx - tp_cx) + tp.get_translation_curve_to_quad_panel()[0],
                   (sc_cy - tp_cy) + tp.get_translation_curve_to_quad_panel()[1]]

    # Curve space visualization
    vertices = []
    for quad in tp.get_quads():
        for edge in quad.get_edges():
            if edge.get_type() in ("seam", "boundary"):
                vertices.append([edge.get_start_vertex_x() + margin, edge.get_start_vertex_y()])
                vertices.append([edge.get_end_vertex_x() + margin, edge.get_end_vertex_y()])
    plot_curve_panel_line(ax_curve, vertices, translation, color='orange', s=2, title=tp.get_name())

    # Quad panel visualization
    quad_vertices = []
    for s_edge, t_edge in cycle.get_st_edge_pairs():
        if t_edge.get_type() in ("seam", "boundary"):
            quad_vertices.append([s_edge.get_start_vertex_quad_x() + margin, s_edge.get_start_vertex_quad_y()])
            quad_vertices.append([s_edge.get_end_vertex_quad_x() + margin, s_edge.get_end_vertex_quad_y()])
    annotation = f"{tp.get_name()}_{source_cycles.index(cycle)}"
    plot_quad_panel_line(ax_quad, quad_vertices, color='orange', s=2, annotation=annotation)

# Final plot settings
ax_curve.set_aspect('equal')
ax_quad.set_aspect('equal')
plt.title("Visualization of Source and Target Panels with Assigned Cycles")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()
