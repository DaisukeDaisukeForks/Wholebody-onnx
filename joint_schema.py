"""Shared anatomical mapping for the compact 73-landmark training targets.

All learned points map to COCO-WholeBody/H3WB labels. H3WB supplies all 73
camera-space 3D targets; ASPset supplies only the 12 common body targets.
Derived references are geometric definitions, not separately labeled surfaces.
"""

BODY_3D_COCO_IDS = list(range(5, 17))
BODY_3D_COMPACT_IDS = list(range(12))
BODY_3D_NAMES = [
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

FINGER_NAMES = [
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "little_mcp", "little_pip", "little_dip", "little_tip",
]
FOOT_NAMES = [
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
]
FACE_68_IDS = [3, 13, 30, 36, 39, 37, 41, 42, 45, 44, 46, 48, 54, 62, 66]
FACE_NAMES = [
    "right_cheek_outline", "left_cheek_outline", "nose_tip",
    "right_eye_outer_corner", "right_eye_inner_corner",
    "right_eye_upper_lid", "right_eye_lower_lid",
    "left_eye_inner_corner", "left_eye_outer_corner",
    "left_eye_upper_lid", "left_eye_lower_lid",
    "mouth_right_corner", "mouth_left_corner", "upper_inner_lip", "lower_inner_lip",
]
COMPACT_COCO_IDS = (
    BODY_3D_COCO_IDS + list(range(92, 112)) + list(range(113, 133))
    + list(range(17, 23)) + [23 + face_id for face_id in FACE_68_IDS]
)
COMPACT_JOINT_NAMES = (
    BODY_3D_NAMES + [f"left_{name}" for name in FINGER_NAMES]
    + [f"right_{name}" for name in FINGER_NAMES] + FOOT_NAMES
    + [f"face_{name}" for name in FACE_NAMES]
)

GROUPS = {
    "body": list(range(0, 12)),
    "left_fingers": list(range(12, 32)),
    "right_fingers": list(range(32, 52)),
    "feet": list(range(52, 58)),
    "face": list(range(58, 73)),
}

SKELETON_EDGES = [
    [0, 1], [0, 2], [2, 4], [1, 3], [3, 5],
    [0, 6], [1, 7], [6, 7], [6, 8], [8, 10], [7, 9], [9, 11],
]
for wrist, start in ((4, 12), (5, 32)):
    for finger in range(5):
        first = start + 4 * finger
        SKELETON_EDGES.extend([[wrist, first], [first, first + 1],
                               [first + 1, first + 2], [first + 2, first + 3]])
    SKELETON_EDGES.extend([[start + 4, start + 8], [start + 8, start + 12],
                           [start + 12, start + 16]])
SKELETON_EDGES.extend([
    [10, 54], [54, 52], [54, 53], [52, 53],
    [11, 57], [57, 55], [57, 56], [55, 56],
    [61, 63], [63, 62], [62, 64], [64, 61],
    [65, 67], [67, 66], [66, 68], [68, 65],
    [69, 71], [71, 70], [70, 72], [72, 69],
])


def _mean_reference(name, ids, meaning):
    return {
        "name": name, "operation": "mean", "joint_ids": ids,
        "required_visible_joint_ids": ids, "coordinate_system": "crop-normalized 2D",
        "meaning": meaning, "learned_extra_point": False,
    }


def _direction(name, start_ids, end_ids, meaning):
    return {
        "name": name, "operation": "mean(end)-mean(start)",
        "start_joint_ids": start_ids, "end_joint_ids": end_ids,
        "required_visible_joint_ids": list(dict.fromkeys(start_ids + end_ids)),
        "coordinate_scale_after_subtraction": [192 / 256, 1.0],
        "coordinate_system": "isotropic 2D crop plane, x right and y down",
        "unit": "crop height; vector is not unit normalized",
        "minimum_vector_length": 1e-6,
        "meaning": meaning, "learned_extra_point": False,
    }


DERIVED_REFERENCES = []
for side, wrist, start, foot_start in (("left", 4, 12, 52), ("right", 5, 32, 55)):
    mcps = [start + 4, start + 8, start + 12, start + 16]
    reference = _mean_reference(
        f"{side}_hand_reference", [wrist] + mcps,
        "Skeleton-derived palm/hand reference from wrist and four non-thumb MCP joints; usable as a hand-back tracking reference, not an independently labeled hand-back surface.",
    )
    reference["aliases"] = [f"{side}_palm_reference", f"{side}_hand_back_reference"]
    DERIVED_REFERENCES.append(reference)
    DERIVED_REFERENCES.append(_direction(
        f"{side}_hand_forward_2d", [wrist], mcps,
        "Wrist-to-MCP-center direction in the image plane.",
    ))
    DERIVED_REFERENCES.append(_direction(
        f"{side}_hand_width_2d", [mcps[0]], [mcps[-1]],
        "Index-MCP to little-MCP direction in the image plane.",
    ))
    DERIVED_REFERENCES.append(_mean_reference(
        f"{side}_toe_center_2d", [foot_start, foot_start + 1],
        "Midpoint of the annotated big-toe and small-toe landmarks.",
    ))
    DERIVED_REFERENCES.append(_direction(
        f"{side}_foot_forward_2d", [foot_start + 2], [foot_start, foot_start + 1],
        "Heel-to-toe-center direction; not a measured 3D sole normal.",
    ))
    DERIVED_REFERENCES.append(_direction(
        f"{side}_toe_span_2d", [foot_start], [foot_start + 1],
        "Big-toe to small-toe direction in the image plane.",
    ))
DERIVED_REFERENCES.extend([
    _mean_reference("right_eye_center_2d", [61, 62], "Midpoint of the two right-eye corners; not a pupil or gaze label."),
    _mean_reference("left_eye_center_2d", [65, 66], "Midpoint of the two left-eye corners; not a pupil or gaze label."),
    _mean_reference("mouth_center_2d", [69, 70], "Midpoint of the two annotated mouth corners."),
])

GEOMETRY2D_NAMES = [
    "left_hand_reference", "right_hand_reference",
    "left_hand_forward_2d", "right_hand_forward_2d",
    "left_foot_forward_2d", "right_foot_forward_2d",
    "left_toe_span_2d", "right_toe_span_2d",
]
_GEOMETRY_TERMS = [
    ([4, 16, 20, 24, 28], [0.2] * 5),
    ([5, 36, 40, 44, 48], [0.2] * 5),
    ([4, 16, 20, 24, 28], [-1.0, 0.25, 0.25, 0.25, 0.25]),
    ([5, 36, 40, 44, 48], [-1.0, 0.25, 0.25, 0.25, 0.25]),
    ([52, 53, 54], [0.5, 0.5, -1.0]),
    ([55, 56, 57], [0.5, 0.5, -1.0]),
    ([52, 53], [-1.0, 1.0]),
    ([55, 56], [-1.0, 1.0]),
]
GEOMETRY2D_WEIGHTS = [[0.0] * len(COMPACT_COCO_IDS) for _ in _GEOMETRY_TERMS]
for row_index, (source_ids, source_weights) in enumerate(_GEOMETRY_TERMS):
    for source_id, source_weight in zip(source_ids, source_weights):
        GEOMETRY2D_WEIGHTS[row_index][source_id] = source_weight

GEOMETRY2D_SCHEMA = {
    "output_name": "geometry2d", "shape": ["batch", 8, 2],
    "confidence_output_name": "geometry_confidence", "confidence_shape": ["batch", 8],
    "names": GEOMETRY2D_NAMES,
    "rows": [
        {"id": index, "name": GEOMETRY2D_NAMES[index],
         "operation": "weighted sum of the selected keypoints, then coordinate scaling",
         "source_compact_ids": ids, "weights": weights,
         "required_visible_joint_ids": ids,
         "coordinate_scale": [1.0, 1.0] if index < 2 else [192 / 256, 1.0],
         "meaning": "crop-normalized hand reference point" if index < 2 else "raw 2D vector in crop-height units, not unit normalized"}
        for index, (ids, weights) in enumerate(_GEOMETRY_TERMS)
    ],
    "confidence": "Minimum source landmark confidence from the pose model. This is not calibrated visibility, a 3D error estimate or a learned surface-classification score.",
    "ground_truth_validity": "All listed source landmarks must have valid 2D labels; absent labels never create a default reference.",
    "direction_validity": "Rows 2 through 7 define an orientation only when their vector norm exceeds 1e-6; a zero vector has no angle even if source confidence is nonzero.",
    "hand_back_reference": "Rows 0 and 1 use the wrist and four non-thumb MCP landmarks. They provide skeleton-based palm/hand-back reference positions, not separately labeled hand-back surface centers.",
    "sole_reference": "Rows 4 through 7 are image-plane foot axes. They do not predict a 3D sole normal or sole surface.",
}

GEOMETRY3D_NAMES = [
    "left_hand_reference_3d", "right_hand_reference_3d",
    "left_hand_forward_3d", "right_hand_forward_3d",
    "left_foot_forward_3d", "right_foot_forward_3d",
    "left_toe_span_3d", "right_toe_span_3d",
]
GEOMETRY3D_WEIGHTS = [row.copy() for row in GEOMETRY2D_WEIGHTS]
GEOMETRY3D_SCHEMA = {
    "output_name": "geometry3d", "shape": ["batch", 8, 3],
    "names": GEOMETRY3D_NAMES, "unit": "meter",
    "coordinates": "camera x right, y down, z away; hip-relative positions and relative vectors",
    "rows": [
        {"id": index, "name": GEOMETRY3D_NAMES[index],
         "operation": "weighted sum of selected 3D landmarks",
         "source_compact_ids": ids, "weights": weights,
         "required_valid_3d_joint_ids": ids,
         "meaning": "hip-relative hand reference position" if index < 2 else "raw 3D vector in meters, not unit normalized"}
        for index, (ids, weights) in enumerate(_GEOMETRY_TERMS)
    ],
    "ground_truth_validity": "All listed source landmarks must have valid 3D labels. H3WB provides these labels; the body-only ASPset cache does not supervise these hand/foot references.",
    "direction_validity": "Rows 2 through 7 define an orientation only for vector norms above 1e-6 meters.",
    "confidence": "Input-landmark confidence does not establish 3D accuracy; no calibrated 3D uncertainty or surface-class score is implied.",
    "hand_back_reference": "Wrist/finger-joint geometry supplies a 3D tracking reference, not a separately annotated dorsal skin point or surface normal.",
    "sole_reference": "Heel/toe geometry supplies 3D foot axes. It does not measure a sole surface, contact state or sole normal.",
}

JOINT_SCHEMA = {
    "schema_id": "compact73-coco-h3wb-v2",
    "version": 2,
    "learned_2d_point_count": len(COMPACT_COCO_IDS),
    "learned_3d_point_count": len(COMPACT_COCO_IDS),
    "source_annotation_point_count": 133,
    "source": "COCO-WholeBody v1.0 2D and H3WB 3D; ASPset common body12",
    "source_ids": COMPACT_COCO_IDS,
    "joint_names": COMPACT_JOINT_NAMES,
    "groups": GROUPS,
    "joints": [
        {"id": index, "name": name, "source_coco_id": source_id,
         "group": next(group for group, members in GROUPS.items() if index in members),
         **({"source_face68_id": source_id - 23} if index >= 58 else {})}
        for index, (name, source_id) in enumerate(zip(COMPACT_JOINT_NAMES, COMPACT_COCO_IDS))
    ],
    "body3d": {
        "compact_2d_ids": BODY_3D_COMPACT_IDS, "source_coco_ids": BODY_3D_COCO_IDS,
        "joint_names": BODY_3D_NAMES, "unit": "meter",
        "coordinates": "camera x right, y down, z away",
        "root": "midpoint of compact joints 6 and 7 (left_hip and right_hip)",
        "role": "Body subset shared with ASPset; the full exported 3D pose has 73 points.",
    },
    "pose3d": {
        "output_name": "keypoints3d", "shape": ["batch", 73, 3],
        "compact_ids": list(range(73)), "source_coco_ids": COMPACT_COCO_IDS,
        "joint_names": COMPACT_JOINT_NAMES, "unit": "meter",
        "coordinates": "camera x right, y down, z away",
        "root": "midpoint of compact joints 6 and 7 (left_hip and right_hip)",
        "h3wb_supervision": "Actual author-released 73 selected multiview reconstructed/completed 3D labels, converted from millimeters to meters.",
        "aspset_supervision": "Only compact0..11 body labels; all other valid3d entries are zero.",
        "input_mask_vs_target_mask": "valid2d describes available 2D inputs; valid3d describes available source 3D labels. Input corruption/dropout does not remove otherwise valid 3D supervision.",
    },
    "input_crop": {
        "height": 256, "width": 192, "aspect": 0.75, "expansion": 1.25,
        "image": "RGB float32 in [0,1] before model preprocessing",
        "xy": "Crop-normalized [0,1] coordinates with a separate validity/confidence channel.",
        "normalization": "The model applies aspect correction and confidence-aware body-root/scale normalization identically at training and inference.",
    },
    "skeleton_edges": SKELETON_EDGES,
    "skeleton_edge_visibility": "Both endpoint labels must be valid; omitted intermediate face landmarks are not synthesized.",
    "duplicate_hand_wrist_handling": {
        "left": {"omitted_source_id": 91, "reference_compact_id": 4, "reference_source_id": 9},
        "right": {"omitted_source_id": 112, "reference_compact_id": 5, "reference_source_id": 10},
    },
    "derived_references": DERIVED_REFERENCES,
    "geometry2d": GEOMETRY2D_SCHEMA,
    "geometry3d": GEOMETRY3D_SCHEMA,
    "derived_visibility": {
        "ground_truth_2d": "All required visible joint IDs must have source visibility > 0 and remain inside the crop.",
        "ground_truth_3d": "All required 3D joint IDs must have valid3d > 0; availability is source-specific and separate from input masking.",
        "inference": "Require all relevant predicted landmarks to pass the caller's confidence criterion; model peak probability is not calibrated visibility.",
        "unavailable": "A missing required point or a direction shorter than minimum_vector_length makes the reference unavailable. No default direction is substituted.",
    },
    "face_semantics": {
        "side_convention": "The person's own anatomical left/right, not the viewer's left/right.",
        "selected_face68_ids": FACE_68_IDS,
        "cheeks": "The two selected points lie on the facial outline and are cheek-outline references, not cheek-surface centers.",
        "eyes": "Corners and eyelids are learned; eye centers are derived. Pupil position and gaze direction are not labeled.",
    },
    "hand_and_foot_semantics": {
        "hand_back": "Hand-back tracking uses the skeleton-derived hand reference. There is no separately supervised dorsal surface point, palm/back classifier, or hand surface normal.",
        "feet": "Learned foot labels are two toe landmarks and a heel per foot in 2D and 3D. Heel/toe axes are derived geometric references; no sole-surface, contact or sole-normal labels are claimed.",
    },
}
