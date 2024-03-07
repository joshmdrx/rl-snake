import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment import SnakeEnvironment

# Initialize the snake environment
env = SnakeEnvironment()
env.reset()
snake_img = env.render()
start_snake_l = len(env._snake)

# Set up the plot
fig, ax = plt.subplots()
img = ax.imshow(snake_img)

# Initialize action
action = {'val': 1}  # Assuming '1' corresponds to an initial valid action
action_dict = {'left': 3, 'up': 0, 'right': 1, 'down': 2}

# Function to handle key presses
def on_key_press(event):
    if event.key in action_dict:
        action['val'] = action_dict[event.key]

fig.canvas.mpl_connect('key_press_event', on_key_press)

# Function to update frame
def update_frame(*args):
    obs, rew, is_over, _ = env.step(action['val'])
    snake_img = env.render()
    img.set_data(snake_img)
    if is_over:
        anim.event_source.stop()
        print('Score:', len(env._snake) - start_snake_l)
    return img,

# Create animation
anim = animation.FuncAnimation(fig, update_frame, interval=100)

plt.show()
