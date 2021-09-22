# Trajectory generation

An API for trajectory generation is provided in [traffic](https://traffic-viz.github.io/index.html). Any object implementing a fit() and sample() methods can be passed as a generation operator.

The following presents the usage of the API on a sample dataset of landing trajectories at Zurich airport.

We define here the reference trajectories that will be used to fit the generation model:

```py
from traffic.core import Traffic

t_reference = Traffic.from_file("path/to/zurich/landing/traffic")
```

You may want to resample those trajectories first so they have the same number of sample points (see [Traffic.resample](https://traffic-viz.github.io/traffic.core.traffic.html#traffic.core.Traffic.resample)). In the case you want to use the track angle as a feature, we advise you to use [Traffic.unwrap()](https://traffic-viz.github.io/traffic.core.traffic.html#traffic.core.Traffic.unwrap).


```py
t_reference = (
    t_reference
    .query("track==track")  # remove any trajectory that misses track values
    .assign_id()
    .resample(30)
    .unwrap()
    .eval(max_workers=4, desc="")
)
```

If relevant you may compute the projection of the longitude and latitude: x and y.

```py
from traffic.core.projection import EuroPP

t_reference = t_reference.compute_xy(projection=EuroPP())
```

The following snippet illustrates the usage of the Generation class to generate `1000` synthetic aircraft trajectories using a GaussianMixture as generation model.

```py
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from traffic.algorithms.generation import Generation

t_generated = Generation(
    generation=GaussianMixture(n_components=1),
    features=["x", "y", "altitude"],
    transform=MinMaxScaler(feature_range=(-1, 1))
).fit(t_reference).sample(n_samples=1000)
```

---

**Note:** the Generation class will automatically add before any fitting, a `timedelta` column to the Traffic DataFrame that will be used as feature. Here is how this parameter is computed:

```py
t_reference = Traffic.from_flights(
    f.assign(
        timedelta=lambda r: (t.timestamp - flight.start).apply(
            lambda t: t.total_seconds()
        )
    )
    for f in t_reference
)
```

`timedelta` will be used to recover correct timestamps for the initialization of Traffic objects.

---------

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
