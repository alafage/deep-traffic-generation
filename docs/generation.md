# Trajectory generation

An API for trajectory generation is provided in `traffic`. Any object implementing a fit() and sample() methods can be passed as a generation operator.


```py
from traffic.core import Traffic
from traffic.algorithms.generation import Generation

# get a reference traffic
t_reference = Traffic.from_file("path/to/traffic")

# generate trajectories
t_generated = Generation(
    generation=GaussianMixture(n_components=1),
    features=["x", "y", "altitude", "timedelta"],
    transform=MinMaxScaler(feature_range=(-1, 1))
).fit(t_reference).sample(n_samples=1000)
```

You can either train a new generation operator with existing traffic or use pre-trained models.

```py
class Generation:
    def __init__(
        self,
        generation: 'GenerationProtocol',
        features: List[str],
        transform: Optional['TransformerProtocol'] = None,
    ) -> None:
        self.generation = generation
        self.features = features
        self.transform = transform

    def fit(self, traffic: Traffic) -> 'Generation':
        """
        Fits a trajectory generation operator based on the traffic data.

        The method:

            - extracts observations of the ``features`` to generate (no default value);
            - *if need be,* apply a transformer to the resulting `X` matrix.
            You may want to consider `MinMaxScaler()
            <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_;
            - returns a Generation object, on which to call fit() or sample() methods. Sampling methods return a Traffic DataFrame containing generated trajectories.
        """
        # features extraction
        X = np.stack(
            list(f.data[self.features].values.ravel() for f in traffic)
        )

        # transforming
        if self.transform is not None:
            X = self.transform.fit_transform(X)

        # fitting
        self.generation.fit(X)

        return self
    
    def sample(self, n_samples: int = 1) -> Traffic:
        """ TODO
        """
        X_hat = self.generation.sample(n_samples)
        X_hat = self.transform.inverse_transform(X_hat)
        t = traffic_from_data(X_hat)

        return t
```
