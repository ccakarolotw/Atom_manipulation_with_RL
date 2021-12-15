from matplotlib import pyplot as plt
from IPython.display import clear_output
import pandas as pd
import matplotlib

def plot_graph(reward_history, avg_reward, ax = None):
    df = pd.DataFrame({'x': range(len(reward_history)), 'Reward': reward_history, 'Average': avg_reward})
    clear_output(wait=True) 
    #plt.style.use('seaborn-darkgrid')

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(df['x'], df['Reward'], marker='', color='silver', linewidth=0.8, alpha=0.9, label='Reward')
    ax.plot(df['x'], df['Average'], marker='', color='chocolate', linewidth=1, alpha=0.9, label='Average')
    #ax.set_ylim([-10,10])
    # plt.legend(loc='upper left')
    ax.set_xlabel("episode", fontsize=12)
    ax.legend()
    plt.show()

def show_reset(img, offset_nm, len_nm, atom_start_position, destination_position, template_nm, template_wh):
    fig, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()

def show_done(img, offset_nm, len_nm, atom_position, atom_start_position, destination_position, template_nm, template_wh, reward, new_destination_absolute_nm = None):
    fig, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    ax.scatter(atom_position[0], atom_position[1], s = 20, linewidths=3, edgecolors='#ff5733', color = None, label='atom')

    if new_destination_absolute_nm is not None:
        ax.scatter(new_destination_absolute_nm[0], new_destination_absolute_nm[1], s = 20, linewidths=3, edgecolors='gray', color = 'gray', label='new destination')

    ax.text(offset_nm[0], offset_nm[1],'reward: {}'.format(reward), ha='center')
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()

def show_step(img, offset_nm, len_nm, start_nm, end_nm, atom_position, atom_start_position, destination_position, template_nm, template_wh, mvolt, pcurrent):
    fig, ax = plt.subplots()
    extent = (offset_nm[0]-0.5*len_nm[0], offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], offset_nm[1])
    ax.imshow(img, extent = extent)
    rect = matplotlib.patches.Rectangle(template_nm,template_wh[0], template_wh[1], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.scatter(atom_start_position[0], atom_start_position[1], s = 20, linewidths=3, edgecolors='#33dbff', color = None, label='start')
    ax.scatter(atom_position[0], atom_position[1], s = 20, linewidths=3, edgecolors='#ff5733', color = None, label='atom')
    ax.scatter(destination_position[0], destination_position[1], s = 20, linewidths=3, edgecolors='#75ff33', color = None, label='gaol')
    ax.arrow(start_nm[0], start_nm[1], (end_nm - start_nm)[0], (end_nm - start_nm)[1],width=0.1, length_includes_head = True)
    ax.text(offset_nm[0]+0.5*len_nm[0], offset_nm[1]+len_nm[0], 'bias(mV):{:.2f}, current(nA):{:.2f}'.format(mvolt,pcurrent/1000))
    plt.legend(frameon=False, labelcolor= 'white')
    plt.show()

