# 사용자가 제공한 코드에서 수정이 필요한 부분만 추출

# add_advanced_features 함수 수정 버전
def add_advanced_features(X_train_feat, X_test_feat, X_train, X_test, events_df, K_val):
    """
    Direction 고도화 실험용 피처 추가
    각 플래그에 따라 하나씩만 추가
    """
    X_train_feat = X_train_feat.copy()
    X_test_feat = X_test_feat.copy()
    
    # 1️⃣ Direction Angle Bin
    # dx, dy 기반 direction angle을 계산하고 여러 구간(bin)으로 나눔
    if USE_DIRECTION_ANGLE_BIN:
        print("  ✓ Direction Angle Bin 피처 추가 중...")
        angle_bin_cols_train = {}
        angle_bin_cols_test = {}
        
        for pos in range(K_val):
            dx_col = f"dx_{pos}"
            dy_col = f"dy_{pos}"
            
            if dx_col in X_train_feat.columns and dy_col in X_train_feat.columns:
                # 각도 계산 (라디안 → 도)
                angle = np.arctan2(X_train_feat[dy_col], X_train_feat[dx_col]) * 180 / np.pi
                # Bin으로 변환
                angle_bin = pd.cut(angle, bins=DIRECTION_ANGLE_BINS, labels=False, include_lowest=True)
                angle_bin_cols_train[f"direction_angle_bin_{pos}"] = angle_bin.fillna(0).astype(int)
            
            if dx_col in X_test_feat.columns and dy_col in X_test_feat.columns:
                angle = np.arctan2(X_test_feat[dy_col], X_test_feat[dx_col]) * 180 / np.pi
                angle_bin = pd.cut(angle, bins=DIRECTION_ANGLE_BINS, labels=False, include_lowest=True)
                angle_bin_cols_test[f"direction_angle_bin_{pos}"] = angle_bin.fillna(0).astype(int)
        
        # 한 번에 병합
        if angle_bin_cols_train:
            angle_bin_df_train = pd.DataFrame(angle_bin_cols_train, index=X_train_feat.index)
            X_train_feat = pd.concat([X_train_feat, angle_bin_df_train], axis=1)
        if angle_bin_cols_test:
            angle_bin_df_test = pd.DataFrame(angle_bin_cols_test, index=X_test_feat.index)
            X_test_feat = pd.concat([X_test_feat, angle_bin_df_test], axis=1)
    
    # 2️⃣ Direction + Distance Bin
    # 패스 거리(distance)를 여러 bin으로 나눔
    if USE_DIRECTION_DISTANCE_BIN:
        print("  ✓ Direction Distance Bin 피처 추가 중...")
        dist_bin_cols_train = {}
        dist_bin_cols_test = {}
        
        for pos in range(K_val):
            dist_col = f"dist_{pos}"
            
            if dist_col in X_train_feat.columns:
                dist_bin = pd.cut(X_train_feat[dist_col], bins=DIRECTION_DISTANCE_BINS, labels=False, include_lowest=True)
                dist_bin_cols_train[f"direction_dist_bin_{pos}"] = dist_bin.fillna(0).astype(int)
            
            if dist_col in X_test_feat.columns:
                dist_bin = pd.cut(X_test_feat[dist_col], bins=DIRECTION_DISTANCE_BINS, labels=False, include_lowest=True)
                dist_bin_cols_test[f"direction_dist_bin_{pos}"] = dist_bin.fillna(0).astype(int)
        
        # 한 번에 병합
        if dist_bin_cols_train:
            dist_bin_df_train = pd.DataFrame(dist_bin_cols_train, index=X_train_feat.index)
            X_train_feat = pd.concat([X_train_feat, dist_bin_df_train], axis=1)
        if dist_bin_cols_test:
            dist_bin_df_test = pd.DataFrame(dist_bin_cols_test, index=X_test_feat.index)
            X_test_feat = pd.concat([X_test_feat, dist_bin_df_test], axis=1)
    
    # 3️⃣ Goal-relative Rotated Direction
    # 상대 골 중심을 기준으로 좌표계를 회전하여 rotated_dx, rotated_dy 생성
    if USE_GOAL_RELATIVE_DIRECTION:
        print("  ✓ Goal-relative Rotated Direction 피처 추가 중...")
        rotated_cols_train = {}
        rotated_cols_test = {}
        
        # 골 중심 좌표
        goal_center_x, goal_center_y = 105, 34
        
        for pos in range(K_val):
            start_x_col = f"start_x_{pos}"
            start_y_col = f"start_y_{pos}"
            dx_col = f"dx_{pos}"
            dy_col = f"dy_{pos}"
            
            if all(col in X_train_feat.columns for col in [start_x_col, start_y_col, dx_col, dy_col]):
                # 시작점을 골 중심 기준으로 이동
                rel_x = X_train_feat[start_x_col] - goal_center_x
                rel_y = X_train_feat[start_y_col] - goal_center_y
                
                # 골 중심을 향하는 각도 계산
                angle_to_goal = np.arctan2(-rel_y, -rel_x)  # 골 방향
                
                # 회전 각도 (골 방향이 x축 양의 방향이 되도록)
                cos_angle = np.cos(angle_to_goal)
                sin_angle = np.sin(angle_to_goal)
                
                # dx, dy를 회전
                rotated_dx = X_train_feat[dx_col] * cos_angle - X_train_feat[dy_col] * sin_angle
                rotated_dy = X_train_feat[dx_col] * sin_angle + X_train_feat[dy_col] * cos_angle
                
                rotated_cols_train[f"rotated_dx_{pos}"] = rotated_dx
                rotated_cols_train[f"rotated_dy_{pos}"] = rotated_dy
            
            if all(col in X_test_feat.columns for col in [start_x_col, start_y_col, dx_col, dy_col]):
                rel_x = X_test_feat[start_x_col] - goal_center_x
                rel_y = X_test_feat[start_y_col] - goal_center_y
                angle_to_goal = np.arctan2(-rel_y, -rel_x)
                cos_angle = np.cos(angle_to_goal)
                sin_angle = np.sin(angle_to_goal)
                rotated_dx = X_test_feat[dx_col] * cos_angle - X_test_feat[dy_col] * sin_angle
                rotated_dy = X_test_feat[dx_col] * sin_angle + X_test_feat[dy_col] * cos_angle
                rotated_cols_test[f"rotated_dx_{pos}"] = rotated_dx
                rotated_cols_test[f"rotated_dy_{pos}"] = rotated_dy
        
        # 한 번에 병합
        if rotated_cols_train:
            rotated_df_train = pd.DataFrame(rotated_cols_train, index=X_train_feat.index)
            X_train_feat = pd.concat([X_train_feat, rotated_df_train], axis=1)
        if rotated_cols_test:
            rotated_df_test = pd.DataFrame(rotated_cols_test, index=X_test_feat.index)
            X_test_feat = pd.concat([X_test_feat, rotated_df_test], axis=1)
    
    return X_train_feat, X_test_feat


