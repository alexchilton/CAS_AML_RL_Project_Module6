import matplotlib.pyplot as plt
import re

def plot_training_metrics(log_content):
    ep_rewards = []
    ep_lengths = []
    value_losses = []
    
    for line in log_content.split('\n'):
        if 'ep_rew_mean' in line:
            match = re.search(r'ep_rew_mean\s*\|\s*([-\d.]+)', line)
            if match:
                ep_rewards.append(float(match.group(1)))
        elif 'ep_len_mean' in line:
            match = re.search(r'ep_len_mean\s*\|\s*([-\d.]+)', line)
            if match:
                ep_lengths.append(float(match.group(1)))
        elif 'value_loss' in line:
            match = re.search(r'value_loss\s*\|\s*([-\d.e+]+\d+)', line)
            if match:
                try:
                    value_losses.append(float(match.group(1)))
                except ValueError:
                    continue

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    
    if ep_rewards:
        ax1.plot(ep_rewards, 'b-')
        ax1.set_title('Average Episode Reward')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
    
    if ep_lengths:
        ax2.plot(ep_lengths, 'g-')
        ax2.set_title('Average Episode Length')
        ax2.set_xlabel('Training Iteration')
        ax2.set_ylabel('Steps')
        ax2.grid(True)
    
    if value_losses:
        ax3.plot(value_losses, 'r-')
        ax3.set_title('Value Loss')
        ax3.set_xlabel('Training Updates')
        ax3.set_ylabel('Loss')
        ax3.grid(True)
    
    plt.tight_layout()
    plt.show()