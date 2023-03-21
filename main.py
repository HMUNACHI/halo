from halo.tuner import tune, generate


model = tune(data_directory="celeba", checkpoint_path="halo_weights")
generate(model=model, n_samples=2, path="tests", extension=".jpg")