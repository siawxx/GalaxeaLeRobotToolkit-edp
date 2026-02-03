# upper body action
EE_POSE_ACTION_LEFT_TOPIC = "/motion_target/target_pose_arm_left"
EE_POSE_ACTION_RIGHT_TOPIC = "/motion_target/target_pose_arm_right"

GRIPPER_ACTION_LEFT_TOPIC = "/motion_target/target_position_gripper_left"
GRIPPER_ACTION_RIGHT_TOPIC = "/motion_target/target_position_gripper_right"

# upper body obs
EE_POSE_OBS_LEFT_TOPIC = "/motion_control/pose_ee_arm_left"
EE_POSE_OBS_RIGHT_TOPIC = "/motion_control/pose_ee_arm_right"

JOINT_OBS_LEFT_TOPIC = "/hdas/feedback_arm_left"
JOINT_OBS_RIGHT_TOPIC = "/hdas/feedback_arm_right"

JOINT_ACTION_LEFT_TOPIC = "/motion_target/target_joint_state_arm_left"
JOINT_ACTION_RIGHT_TOPIC = "/motion_target/target_joint_state_arm_right"

GRIPPER_OBS_LEFT_TOPIC = "/hdas/feedback_gripper_left"
GRIPPER_OBS_RIGHT_TOPIC = "/hdas/feedback_gripper_right"

RGB_HEAD_LEFT_TOPIC = "/hdas/camera_head/left_raw/image_raw_color/compressed"
RGB_HEAD_RIGHT_TOPIC = "/hdas/camera_head/right_raw/image_raw_color/compressed"
RGB_WRIST_LEFT_TOPIC = "/hdas/camera_wrist_left/color/image_raw/compressed"
RGB_WRIST_RIGHT_TOPIC = "/hdas/camera_wrist_right/color/image_raw/compressed"

DEPTH_HEAD_TOPIC = "/hdas/camera_head/depth/depth_registered"
DEPTH_LEFT_TOPIC = "/hdas/camera_wrist_left/aligned_depth_to_color/image_raw"
DEPTH_RIGHT_TOPIC = "/hdas/camera_wrist_right/aligned_depth_to_color/image_raw"

# lower body action
CHASSIS_ACTION_TOPIC = "/motion_target/target_speed_chassis"
TORSO_ACTION_TOPIC = "/motion_target/target_joint_state_torso"
TORSO_ACTION_SPEED_TOPIC = "/motion_target/target_speed_torso"
# lower body obs
CHASSIS_OBS_TOPIC = "/hdas/feedback_chassis"
CHASSIS_IMU_TOPIC = "/hdas/imu_chassis"
TORSO_OBS_TOPIC = "/hdas/feedback_torso"

# motion control topics
JOINT_CONTROL_ACTION_LEFT_TOPIC = "/motion_control/control_arm_left"
JOINT_CONTROL_ACTION_RIGHT_TOPIC = "/motion_control/control_arm_right"

GRIPPER_CONTROL_ACTION_LEFT_TOPIC = "/motion_control/control_gripper_left"
GRIPPER_CONTROL_ACTION_RIGHT_TOPIC = "/motion_control/control_gripper_right"

CHASSIS_CONTROL_ACTION_TOPIC = "/motion_control/control_chassis"
TORSO_CONTROL_ACTION_TOPIC = "/motion_control/control_torso"

