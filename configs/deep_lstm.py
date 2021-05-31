_base_ = ['_base_/imagenet.py']

attributer = dict(
    layer='rnn_4',
    classifier=dict(
        type='vgg16',
        pretrained=True),
    iba=dict(
        input_or_output="output",
        active_neurons_threshold=0.01,
        initial_alpha=5.0),
    img_iba=dict(
        initial_alpha=5.0,
        sigma=0.0,
    )
)

estimation_cfg = dict(
    n_samples=1000,
    progbar=True,
)

attribution_cfg = dict(
    iba=dict(
        batch_size=10,
        beta=0.5),
    gan=dict(
        dataset_size=200,
        sub_dataset_size=20,
        lr=0.00005,
        batch_size=32,
        weight_clip=0.01,
        epochs=20,
        critic_iter=5),
    img_iba=dict(
        beta=0.1,
        opt_steps=30,
        lr=0.5,
        batch_size=10),
    feat_mask=dict(
        upscale=True,
        show=False),
    img_mask=dict(
        show=False)
)


sanity_check = dict(
    perturb_layers=[
        'classifier.6',
        'classifier.0',
        'features.21',
        'features.17',
        'features.7',
        'features.0'],
    check='img_iba')