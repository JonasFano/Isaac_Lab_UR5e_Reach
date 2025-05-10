import math

class TaskParams:
    #################################
    ### General Simulation Params ###
    #################################
    decimation = 2
    episode_length_s = 15.0
    dt = 1/100
    render_interval = 2


    #######################
    ### Differential IK ###
    #######################
    command_type = "pose"
    use_relative_mode = True
    ik_method = "dls"
    action_scale = 0.05


    ##############
    ### Reward ###
    ##############
    end_effector_position_tracking_weight = -0.2

    end_effector_position_tracking_fine_grained_std = 0.1
    end_effector_position_tracking_fine_grained_weight = 0.1

    end_effector_orientation_tracking_weight = -0.1

    action_rate_weight = -1.0 #-1e-4
    action_rate_curriculum_weight = -1.0
    # action_magnitude_weight = -1e-4
    # action_magnitude_curriculum_weight = -0.02
    # ee_acc_weight = -1e-4
    # ee_acc_curriculum_weight = -0.001
    curriculum_num_steps = 16000


    #############
    ### Robot ###
    #############
    # Robot parameters/gains
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    ee_body_name = "wrist_3_link"
    robot_base_init_position = (0.3, -0.1, 0.0)

    robot_vel_limit = 180.0
    robot_effort_limit = 87.0
    robot_stiffness = 10000000.0

    shoulder_pan_mass = 3.761
    shoulder_lift_mass = 8.058
    elbow_mass = 2.846
    wrist_1_mass = 1.37
    wrist_2_mass = 1.3
    wrist_3_mass = 0.365

    # Critically damped damping 
    shoulder_pan_damping = 2 * math.sqrt(robot_stiffness * shoulder_pan_mass) 
    shoulder_lift_damping = 2 * math.sqrt(robot_stiffness * shoulder_lift_mass)
    elbow_damping = 2 * math.sqrt(robot_stiffness * elbow_mass)
    wrist_1_damping = 2 * math.sqrt(robot_stiffness * wrist_1_mass)
    wrist_2_damping = 2 * math.sqrt(robot_stiffness * wrist_2_mass)
    wrist_3_damping = 2 * math.sqrt(robot_stiffness * wrist_3_mass)
    
    # Domain randomize robot stiffness and damping
    robot_randomize_stiffness = (0.5, 1.5)
    robot_randomize_damping = (0.5, 1.5)
    robot_randomize_stiffness_operation = "scale"
    robot_randomize_damping_operation = "scale"
    robot_randomize_stiffness_distribution = "uniform"
    robot_randomize_damping_distribution = "uniform"

    robot_initial_joint_pos = [1.3, -2.0, 2.0, -1.5, -1.5, 0.0, 0.0, 0.0] # With gripper joint pos set to 0.0
    robot_reset_joints_pos_range = (0.7, 1.3)
    robot_reset_joints_vel_range = (0.0, 0.0)

    gripper_offset = [0.0, 0.0, 0.0] # Set to [0.0, 0.0, 0.15] for Hand E Gripper


    ###############
    ### Command ###
    ###############
    resampling_time_range = (5.0, 5.0)
    visualize_frame = True

    sample_range_pos_x = (-0.15, 0.15)
    sample_range_pos_y = (0.25, 0.5)
    sample_range_pos_z = (0.2, 0.5)
    sample_range_roll = (0.0, 0.0)
    sample_range_pitch = (math.pi, math.pi) # depends on end-effector axis
    sample_range_yaw = (-3.14, 3.14)