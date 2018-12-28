import matplotlib.pyplot as plt
plt.style.use("dark_background")

def plot_image(x, idx):
    """
    """
    picture = x[idx].squeeze().cpu().numpy()
    fig, axes = plt.subplots(1,1, figsize=(10,10))
    axes.imshow(picture, cmap="gray")

def plot_confusion(mat):
    """
    """
    fig, axes = plt.subplots(1,1, figsize=(10,10))
    axes.imshow(mat, cmap="gray")

def plot_training(traccs, losses, valaccs, vallosses):
    """
    """
    fig, axes = plt.subplots(2,2, figsize=(10,10))
    axes[0,0].plot(traccs)
    axes[0,1].plot(losses)
    axes[1,0].plot(valaccs)
    axes[1,1].plot(vallosses)
