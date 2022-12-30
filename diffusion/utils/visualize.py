from torch import flip
import matplotlib.pyplot as plt


def get_display_image(image, mean=0., std=1.):
    return (image.permute(1, 2, 0) * std + mean).clamp(0, 1).numpy()


def display_image(image, **kwargs):
    plt.imshow(get_display_image(image, **kwargs))
    plt.show()


def visualize_diffusions(samples_a, samples_b, num_samples, num_steps):
    # Assumes samples_a is shape B x C x H x W x T
    total_steps = samples_a.shape[-1]
    fig, axis = plt.subplots(num_steps + 1, num_samples * 2, figsize=(num_samples * 2, num_steps))
    for sample_ix in range(num_samples):
        for plot_ix, step_ix in enumerate(range(0, total_steps, total_steps // num_steps)):
            axis[plot_ix, sample_ix].set_axis_off()
            axis[plot_ix, sample_ix + num_samples].set_axis_off()
            axis[plot_ix, sample_ix].imshow(
                get_display_image(flip(samples_a, [-1])[sample_ix, :, :, :, step_ix])
            )
            axis[plot_ix, sample_ix + num_samples].imshow(
                get_display_image(flip(samples_b, [-1])[sample_ix, :, :, :, step_ix])
            )