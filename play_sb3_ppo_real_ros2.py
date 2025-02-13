import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
import numpy as np
import torch
from stable_baselines3 import PPO
import random
from scipy.spatial.transform import Rotation as R

# Load the pre-trained model
checkpoint_path = "/home/jofa/Downloads/Repositories/Isaac_Lab_UR5e_Reach/models/nqblneg1/model.zip"
print(f"Loading checkpoint from: {checkpoint_path}")
agent = PPO.load(checkpoint_path)

# Random Pose Sampling Function
def sample_random_pose():
    """Randomly sample a target pose within specified bounds."""
    pos_x = random.uniform(-0.2, 0.2)
    pos_y = random.uniform(0.35, 0.55)
    pos_z = random.uniform(0.15, 0.4)
    roll = 0.0
    pitch = np.pi  # End-effector pointing down
    yaw = random.uniform(-np.pi, np.pi)
    return [pos_x, pos_y, pos_z, roll, pitch, yaw]

# Convert pose to ROS 2 message
def pose_to_ros_message(pose):
    """Convert a TCP pose to a ROS 2 Pose message."""
    ros_pose = Pose()
    ros_pose.position.x = pose[0]
    ros_pose.position.y = pose[1]
    ros_pose.position.z = pose[2]
    rotation = R.from_euler('xyz', pose[3:]).as_quat()
    ros_pose.orientation.x = rotation[0]
    ros_pose.orientation.y = rotation[1]
    ros_pose.orientation.z = rotation[2]
    ros_pose.orientation.w = rotation[3]
    return ros_pose

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('robot_control_node')
        self.publisher_ = self.create_publisher(Pose, 'tcp_target_pose', 10)
        self.subscription = self.create_subscription(Pose, 'tcp_current_pose', self.tcp_pose_callback, 10)
        self.current_tcp_pose = None
        self.previous_actions = np.zeros(6)
        self.target_pose = sample_random_pose()
        self.get_logger().info(f"Target Pose: {self.target_pose}")
        self.timer = self.create_timer(0.1, self.control_loop)

    def tcp_pose_callback(self, msg):
        """Callback to update the current TCP pose."""
        self.current_tcp_pose = [
            msg.position.x,
            msg.position.y,
            msg.position.z,
            *R.from_quat([
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ]).as_rotvec()
        ]

    def control_loop(self):
        if self.current_tcp_pose is None:
            self.get_logger().warn("Waiting for current TCP pose...")
            return

        obs = np.concatenate((self.current_tcp_pose, self.target_pose, self.previous_actions))

        with torch.inference_mode():
            action, _ = agent.predict(obs, deterministic=True)

        self.previous_actions = action  # Save the action

        # Calculate the new TCP pose based on the action
        new_tcp = np.array(self.current_tcp_pose)
        new_tcp[:3] += action[:3]

        current_rotation = R.from_rotvec(new_tcp[3:])
        displacement_rotation = R.from_rotvec(action[3:])
        new_rotation = current_rotation * displacement_rotation
        new_tcp[3:] = new_rotation.as_rotvec()

        # Publish the new target pose
        self.publisher_.publish(pose_to_ros_message(new_tcp))

        # Check if target pose is reached
        distance = np.linalg.norm(np.array(self.target_pose[:3]) - np.array(self.current_tcp_pose[:3]))
        if distance < 0.02:
            self.get_logger().info("Target pose reached. Sampling new pose.")
            self.target_pose = sample_random_pose()
            self.get_logger().info(f"New Target Pose: {self.target_pose}")

# Main function to run the ROS 2 node
def main(args=None):
    rclpy.init(args=args)
    robot_control_node = RobotControlNode()

    try:
        rclpy.spin(robot_control_node)
    except KeyboardInterrupt:
        robot_control_node.get_logger().info("Stopping robot execution.")
    finally:
        robot_control_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
