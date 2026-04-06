# TODO
- Add a description about platform / backend limitation (i.e., Linux is required for torch.compile and Jax with GPU) in README
- Clean up the runs declaration in the `plot_benchmark_figures.ipynb` if possible
- sns.replot should only sort by hue, set style=None otherwise the sorting and color can get silently wrong
- Improve the color palette for visibility
- merge plot_packages with packages_color dicts would be cleaner
- Run the updated phaser on cloud machines so we can update `plot_device_performance.ipynb` as well