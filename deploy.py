import time
import argparse
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name")
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Load config parameters
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]
        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]
        
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
        
        urdf_to_usd = config["urdf_to_usd"]
        usd_to_urdf = config["usd_to_urdf"]

        
        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
        
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        slow_motion_factor = config.get("slow_motion_factor", 1.0)

    # Initialize variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    prev_action = np.zeros(num_actions, dtype=np.float32)  # Initialize previous action as zeros
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    
    # Demo variables
    curr_demo_dof_pos = np.array([0.3, 0.3, 0.25, -0.25, 0.0, 0.0, 0.97, 0.97, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    next_demo_dof_pos = np.array([0.3, 0.3, 0.25, -0.25, 0.0, 0.0, 0.97, 0.97, 0.15, 0.15, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    single_obs_dim = 124  # 3+3+3+29+29+29+14+14

    # Load MuJoCo model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    policy = torch.jit.load(policy_path)

    # Set initial pose
    d.qpos[7:] = default_angles  # 设置关节初始位置
    d.qpos[2] = 0.79  # 设置机身初始高度
    mujoco.mj_forward(m, d)  # 前向运动学更新

    # Global variable for pause state
    paused = True
    
    def key_callback(keycode):
        global paused
        if keycode == 32:  # Space key
            paused = not paused
            if not paused:
                print("Simulation started - press SPACE again to pause")
            else:
                print("Simulation paused - press SPACE to continue")
        elif keycode == 256:  # ESC key
            print("Use Ctrl+C to quit")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        print("=== Controls ===")
        print("SPACE: Start/Pause simulation")
        print("ESC: Show quit message")
        print("Focus on the MuJoCo window and press SPACE to start")
        print("================")
        
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            
            # Only run simulation if not paused
            if not paused:
                # PD control
                tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
                d.ctrl[:] = tau
                
                mujoco.mj_step(m, d)
                counter += 1
            
            if not paused and counter % control_decimation == 0:
                # Create observation
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

            
                qj = (qj - default_angles) * dof_pos_scale
                qj = qj[urdf_to_usd]

                dqj = dqj * dof_vel_scale
                dqj = dqj[urdf_to_usd]
                
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                # Build current observation vector
                obs[:3] = omega
                obs[3:6] = gravity_orientation
                obs[6:9] = cmd * cmd_scale
                obs[9 : 38] = qj
                obs[38 : 67] = dqj
                obs[67 : 96] = prev_action
                
                # Add demo data to observation
                obs[96 : 110] = curr_demo_dof_pos
                obs[110 : 124] = next_demo_dof_pos
                

                
                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                action = policy(obs_tensor).detach().numpy().squeeze()
                prev_action = action
                action = action[usd_to_urdf]
                target_dof_pos = action * action_scale + default_angles
                

                
            viewer.sync()
            
            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step * slow_motion_factor)
            else:
                # Even if we're running slow, add some delay for slow motion
                time.sleep(m.opt.timestep * (slow_motion_factor - 1.0))

                