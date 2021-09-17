# Trajectory clustering

```py

    from traffic.algorithms.generation import generate, load_model

    model = load_model("TCVAE").from_checkpoint("path/to/folder")

    t_generated = generate(
        n_samples: int,
        model: GenerationProtocol,
        reference=None,
    )


    class Generation:

        def __init__(self, model: LightningModule) -> None:
            self.model = model

        @classmethod
        def load_model(clf, )
```