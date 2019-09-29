from bfgn.architectures import alex_net, dense_flat_net, dense_unet, dilation_net, flat_net, residual_dilation_net, \
    residual_flat_net, residual_unet, unet


def test_alex_net_builds_and_compiles():
    model = alex_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        pool_size=(2, 2),
        use_batch_norm=True,
        use_growth=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_dense_flat_net_builds_and_compiles():
    model = dense_flat_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        use_batch_norm=True,
        use_growth=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_dense_unet_builds_and_compiles():
    model = dense_unet.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        pool_size=(2, 2),
        use_batch_norm=True,
        use_growth=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_dilation_net_builds_and_compiles():
    model = dilation_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        dilation_rate=2,
        filters=3,
        kernel_size=(3, 3),
        num_layers=4,
        padding="same",
        use_batch_norm=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_flat_net_builds_and_compiles():
    model = flat_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        filters=3,
        kernel_size=(3, 3),
        num_layers=4,
        padding="same",
        use_batch_norm=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_residual_dilation_net_builds_and_compiles():
    model = residual_dilation_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        use_batch_norm=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_residual_flat_net_builds_and_compiles():
    model = residual_flat_net.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        use_batch_norm=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_residual_unet_builds_and_compiles():
    model = residual_unet.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        pool_size=(2, 2),
        use_batch_norm=True,
        use_growth=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")


def test_unet_builds_and_compiles():
    model = unet.create_model(
        inshape=(16, 16, 2),
        n_classes=2,
        output_activation="softmax",
        block_structure=(1, 2, 1),
        filters=3,
        kernel_size=(3, 3),
        padding="same",
        pool_size=(2, 2),
        use_batch_norm=True,
        internal_activation="relu",
        use_growth=True,
        use_initial_colorspace_transformation_layer=True,
    )
    model.compile(optimizer="adam", loss="mse")
