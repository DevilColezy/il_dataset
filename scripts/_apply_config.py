"""Apply targeted edits to il_dataset_config.yaml."""
import sys

with open('config/il_dataset_config.yaml', 'r', encoding='utf-8') as f:
    content = f.read()

edits = 0

# ── 1. Add distance_bands, gap_constraints, planner_clearance ──
old = """      density_tier_thresholds:
        sparse_max: 0.06
        medium_max: 0.14

    # -- Execution control"""
new = """      density_tier_thresholds:
        sparse_max: 0.06
        medium_max: 0.14

    # -- Distance bands (adaptive obstacle regions) --
    distance_bands:
      micro:
        flight_min_m:  1.0
        flight_max_m:  5.0
        region_x_half: 2.0
        region_y_pad:  1.0
        region_z_pad:  1.0
      short:
        flight_min_m:  5.0
        flight_max_m: 10.0
        region_x_half: 3.0
        region_y_pad:  1.5
        region_z_pad:  1.5
      medium:
        flight_min_m: 10.0
        flight_max_m: 18.0
        region_x_half: 4.0
        region_y_pad:  2.0
        region_z_pad:  1.5
      long:
        flight_min_m: 18.0
        flight_max_m: 25.0
        region_x_half: 5.5
        region_y_pad:  2.5
        region_z_pad:  2.0
      extreme:
        flight_min_m: 25.0
        flight_max_m: 32.0
        region_x_half: 7.0
        region_y_pad:  3.0
        region_z_pad:  2.0

    # -- Per-density gap constraints --
    gap_constraints_by_density:
      sparse:
        minimum_surface_gap_m: 1.50
        minimum_post_inflation_gap_m: 0.50
      medium:
        minimum_surface_gap_m: 1.00
        minimum_post_inflation_gap_m: 0.35
      dense:
        minimum_surface_gap_m: 0.60
        minimum_post_inflation_gap_m: 0.20

    # -- Per-density planner clearance --
    planner_clearance_by_density:
      sparse:
        target_clearance: 0.25
        min_clearance: 0.05
      medium:
        target_clearance: 0.20
        min_clearance: 0.03
      dense:
        target_clearance: 0.15
        min_clearance: 0.02

    # -- Execution control"""
if old in content:
    content = content.replace(old, new)
    edits += 1
    print("Edit 1 OK: distance_bands + gap + clearance added")
else:
    print("ERROR: Edit 1 not found!")

# ── 2. Update task generation settings ──
old = """      # Keep both straight-flight and avoidance samples.
      require_direct_path_blocked: false
      minimum_direct_blocker_count: 1
      maximum_direct_blocker_count: 3

      require_astar_reachable: true
      minimum_detour_ratio: 1.0         # no minimum
      maximum_detour_ratio: 1000.0      # effectively unlimited"""
new = """      # All tasks must be blocked to maximise Guide-label diversity.
      require_direct_path_blocked: true
      minimum_direct_blocker_count: 1
      maximum_direct_blocker_count: 3
      direct_path_corridor_radius_m: 0.40

      require_astar_reachable: true
      minimum_detour_ratio: 1.25
      maximum_detour_ratio: 3.50

      # Adaptive obstacle region: size the XY region to the distance band
      use_adaptive_region: true"""
if old in content:
    content = content.replace(old, new)
    edits += 1
    print("Edit 2 OK: task generation settings updated")
else:
    print("ERROR: Edit 2 not found!")

# ── 3. Update task_type_weights ──
old = """        task_type_weights_by_density_tier:
          default:
            {clear: 0.30, single_left: 0.20, single_right: 0.20,
             single_center: 0.15, multi_blocker: 0.15}
          sparse:
            {clear: 0.55, single_left: 0.15, single_right: 0.15,
             single_center: 0.15, multi_blocker: 0.00}
          medium:
            {clear: 0.20, single_left: 0.20, single_right: 0.20,
             single_center: 0.20, multi_blocker: 0.20}
          dense:
            {clear: 0.00, single_left: 0.15, single_right: 0.15,
             single_center: 0.10, multi_blocker: 0.60}"""
new = """        task_type_weights_by_density_tier:
          default:
            {clear: 0.00, single_left: 0.25, single_right: 0.25,
             single_center: 0.20, multi_blocker: 0.30}
          sparse:
            {clear: 0.00, single_left: 0.30, single_right: 0.30,
             single_center: 0.25, multi_blocker: 0.15}
          medium:
            {clear: 0.00, single_left: 0.22, single_right: 0.22,
             single_center: 0.16, multi_blocker: 0.40}
          dense:
            {clear: 0.00, single_left: 0.10, single_right: 0.10,
             single_center: 0.05, multi_blocker: 0.75}"""
if old in content:
    content = content.replace(old, new)
    edits += 1
    print("Edit 3 OK: task_type_weights updated")
else:
    print("ERROR: Edit 3 not found!")

# ── 4. Update blocker_distance_weights ──
old = """        blocker_distance_weights_by_density_tier:
          sparse: {near: 0.35, middle: 0.40, far: 0.25}
          medium: {near: 0.45, middle: 0.45, far: 0.10}
          dense: {near: 0.60, middle: 0.40, far: 0.00}"""
new = """        blocker_distance_weights_by_density_tier:
          sparse: {near: 0.65, middle: 0.30, far: 0.05}
          medium: {near: 0.70, middle: 0.25, far: 0.05}
          dense:  {near: 0.80, middle: 0.20, far: 0.00}"""
if old in content:
    content = content.replace(old, new)
    edits += 1
    print("Edit 4 OK: blocker_distance_weights updated")
else:
    print("ERROR: Edit 4 not found!")

# ── 5. Replace profiles ──
profiles_start_marker = "    profiles:"
idx = content.index(profiles_start_marker)
# Find the next section after profiles
next_section_marker = "\n  # "
next_idx = content.index(next_section_marker, idx + len(profiles_start_marker) + 50)

new_profiles = """    profiles:
      # ──────────────────────────────────────────────────────────────
      #  MICRO  (1-5 m)  -- pure reactive avoidance
      # ──────────────────────────────────────────────────────────────
      - name: "M01_micro_medium"
        enabled: true
        scene_count: 2
        seed_offset: 1010
        distance_band: micro
        density_min: 0.04
        density_max: 0.07
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.40
            radius_max_m: 0.60
            consecutive_fail_threshold: 40
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.25
            radius_max_m: 0.40
            consecutive_fail_threshold: 80
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.10
            radius_max_m: 0.25
            consecutive_fail_threshold: 150

      - name: "M02_micro_dense"
        enabled: true
        scene_count: 3
        seed_offset: 1020
        distance_band: micro
        density_min: 0.10
        density_max: 0.15
        size_groups:
          large:
            capacity_fraction: 0.25
            radius_min_m: 0.40
            radius_max_m: 0.60
            consecutive_fail_threshold: 50
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.25
            radius_max_m: 0.40
            consecutive_fail_threshold: 100
          small:
            capacity_fraction: 0.40
            radius_min_m: 0.10
            radius_max_m: 0.25
            consecutive_fail_threshold: 180

      # ──────────────────────────────────────────────────────────────
      #  SHORT  (5-10 m)
      # ──────────────────────────────────────────────────────────────
      - name: "S01_short_sparse"
        enabled: true
        scene_count: 1
        seed_offset: 2010
        distance_band: short
        density_min: 0.015
        density_max: 0.030
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.50
            radius_max_m: 0.80
            consecutive_fail_threshold: 30
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.30
            radius_max_m: 0.50
            consecutive_fail_threshold: 60
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.15
            radius_max_m: 0.30
            consecutive_fail_threshold: 100

      - name: "S02_short_medium"
        enabled: true
        scene_count: 2
        seed_offset: 2020
        distance_band: short
        density_min: 0.04
        density_max: 0.07
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.50
            radius_max_m: 0.80
            consecutive_fail_threshold: 40
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.30
            radius_max_m: 0.50
            consecutive_fail_threshold: 80
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.15
            radius_max_m: 0.30
            consecutive_fail_threshold: 120

      - name: "S03_short_dense"
        enabled: true
        scene_count: 4
        seed_offset: 2030
        distance_band: short
        density_min: 0.10
        density_max: 0.15
        size_groups:
          large:
            capacity_fraction: 0.25
            radius_min_m: 0.50
            radius_max_m: 0.80
            consecutive_fail_threshold: 50
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.30
            radius_max_m: 0.50
            consecutive_fail_threshold: 100
          small:
            capacity_fraction: 0.40
            radius_min_m: 0.15
            radius_max_m: 0.30
            consecutive_fail_threshold: 150

      # ──────────────────────────────────────────────────────────────
      #  MEDIUM  (10-18 m)  * core training band
      # ──────────────────────────────────────────────────────────────
      - name: "M01_medium_sparse"
        enabled: true
        scene_count: 3
        seed_offset: 3010
        distance_band: medium
        density_min: 0.015
        density_max: 0.030
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.60
            radius_max_m: 1.00
            consecutive_fail_threshold: 30
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.35
            radius_max_m: 0.60
            consecutive_fail_threshold: 60
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.15
            radius_max_m: 0.35
            consecutive_fail_threshold: 100

      - name: "M02_medium_medium"
        enabled: true
        scene_count: 6
        seed_offset: 3020
        distance_band: medium
        density_min: 0.04
        density_max: 0.07
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.60
            radius_max_m: 1.00
            consecutive_fail_threshold: 40
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.35
            radius_max_m: 0.60
            consecutive_fail_threshold: 80
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.15
            radius_max_m: 0.35
            consecutive_fail_threshold: 120

      - name: "M03_medium_dense"
        enabled: true
        scene_count: 8
        seed_offset: 3030
        distance_band: medium
        density_min: 0.10
        density_max: 0.15
        size_groups:
          large:
            capacity_fraction: 0.25
            radius_min_m: 0.60
            radius_max_m: 1.00
            consecutive_fail_threshold: 50
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.35
            radius_max_m: 0.60
            consecutive_fail_threshold: 100
          small:
            capacity_fraction: 0.40
            radius_min_m: 0.15
            radius_max_m: 0.35
            consecutive_fail_threshold: 150

      # ──────────────────────────────────────────────────────────────
      #  LONG  (18-25 m)
      # ──────────────────────────────────────────────────────────────
      - name: "L01_long_sparse"
        enabled: true
        scene_count: 3
        seed_offset: 4010
        distance_band: long
        density_min: 0.015
        density_max: 0.030
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.80
            radius_max_m: 1.50
            consecutive_fail_threshold: 30
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.45
            radius_max_m: 0.80
            consecutive_fail_threshold: 60
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.20
            radius_max_m: 0.45
            consecutive_fail_threshold: 100

      - name: "L02_long_medium"
        enabled: true
        scene_count: 4
        seed_offset: 4020
        distance_band: long
        density_min: 0.04
        density_max: 0.07
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 0.80
            radius_max_m: 1.50
            consecutive_fail_threshold: 40
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.45
            radius_max_m: 0.80
            consecutive_fail_threshold: 80
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.20
            radius_max_m: 0.45
            consecutive_fail_threshold: 120

      - name: "L03_long_dense"
        enabled: true
        scene_count: 5
        seed_offset: 4030
        distance_band: long
        density_min: 0.10
        density_max: 0.15
        size_groups:
          large:
            capacity_fraction: 0.25
            radius_min_m: 0.80
            radius_max_m: 1.50
            consecutive_fail_threshold: 50
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.45
            radius_max_m: 0.80
            consecutive_fail_threshold: 100
          small:
            capacity_fraction: 0.40
            radius_min_m: 0.20
            radius_max_m: 0.45
            consecutive_fail_threshold: 150

      # ──────────────────────────────────────────────────────────────
      #  EXTREME  (25-32 m)
      # ──────────────────────────────────────────────────────────────
      - name: "E01_extreme_sparse"
        enabled: true
        scene_count: 2
        seed_offset: 5010
        distance_band: extreme
        density_min: 0.015
        density_max: 0.030
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 1.00
            radius_max_m: 2.00
            consecutive_fail_threshold: 30
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.55
            radius_max_m: 1.00
            consecutive_fail_threshold: 60
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.25
            radius_max_m: 0.55
            consecutive_fail_threshold: 100

      - name: "E02_extreme_medium"
        enabled: true
        scene_count: 2
        seed_offset: 5020
        distance_band: extreme
        density_min: 0.04
        density_max: 0.07
        size_groups:
          large:
            capacity_fraction: 0.30
            radius_min_m: 1.00
            radius_max_m: 2.00
            consecutive_fail_threshold: 40
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.55
            radius_max_m: 1.00
            consecutive_fail_threshold: 80
          small:
            capacity_fraction: 0.35
            radius_min_m: 0.25
            radius_max_m: 0.55
            consecutive_fail_threshold: 120

      - name: "E03_extreme_dense"
        enabled: true
        scene_count: 3
        seed_offset: 5030
        distance_band: extreme
        density_min: 0.10
        density_max: 0.15
        size_groups:
          large:
            capacity_fraction: 0.25
            radius_min_m: 1.00
            radius_max_m: 2.00
            consecutive_fail_threshold: 50
          medium:
            capacity_fraction: 0.35
            radius_min_m: 0.55
            radius_max_m: 1.00
            consecutive_fail_threshold: 100
          small:
            capacity_fraction: 0.40
            radius_min_m: 0.25
            radius_max_m: 0.55
            consecutive_fail_threshold: 150
"""
content = content[:idx] + new_profiles + content[next_idx:]
edits += 1
print("Edit 5 OK: profiles replaced (15 distance-band x density profiles)")

with open('config/il_dataset_config.yaml', 'w', encoding='utf-8') as f:
    f.write(content)
print("\nAll {} edits applied!".format(edits))
