class PoseDefinition:
    part_id_dict = {
        # 'nose_id': [0],
        'left_eye_id': [1, 2, 3],
        'right_eye_id': [4, 5, 6],
        'mouse_id_id': [9, 10],
        'shoulder_id': [11, 12],
        'left_wrist_id': [12, 14, 16],
        'right_wrist_id': [11, 13, 15],
        'left_hand_id': [16, 18, 20, 22, 16],
        'right_hand_id': [15, 17, 19, 21, 15],
        'body_id': [11, 12, 24, 23, 11],
        'left_leg_id': [23, 25, 27],
        'right_leg_id': [24, 26, 28],
        'left_foot_id': [27, 29, 31, 27],
        'right_foot_id': [28, 30, 32, 28]
    }

    ALL_ID_LINKs = list(part_id_dict.values())

    EDGES = []
    for i in [[[e, ID_LINK[i + 1]] for i, e in enumerate(ID_LINK) if i < len(ID_LINK)-1] for ID_LINK in ALL_ID_LINKs]:
        for j in i:
            EDGES.append(j)
