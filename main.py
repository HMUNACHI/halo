from halo.tuner import tune, generate


model = tune("celeba", "tests")
generate(model=model, n_samples=2, path="my_path", extension=".jpg")