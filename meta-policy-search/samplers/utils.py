import numpy as np
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True, video_filename='sim_out.mp4', ignore_done=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    images = []

    ''' get wrapped env '''
    wrapped_env = env
    while hasattr(wrapped_env, '_wrapped_env'):
        wrapped_env = wrapped_env._wrapped_env

    frame_skip = wrapped_env.frame_skip if hasattr(wrapped_env, 'frame_skip') else 1
    assert hasattr(wrapped_env, 'dt'), 'environment must have dt attribute that specifies the timestep'
    timestep = wrapped_env.dt

    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action([o])
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d and not ignore_done: # and not animated:
            break
        o = next_o
        if animated:
            env.render()
            time.sleep(timestep*frame_skip / speedup)
            if save_video:
                from PIL import Image
                image = env.wrapped_env.wrapped_env.get_viewer().get_image()
                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                images.append(np.flipud(np.array(pil_image)))

    if animated:
        if save_video:
            import moviepy.editor as mpy
            fps = int(speedup/timestep * frame_skip)
            clip = mpy.ImageSequenceClip(images, fps=fps)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename, fps=fps)
            else:
                clip.write_videofile(video_filename, fps=fps)
        #return

    return dict(
        observations=observations,
        actons=actions,
        rewards=rewards,
        agent_infos=agent_infos,
        env_infos=env_infos
        )