import pytest
from traffic.core import Traffic
from traffic.core.projection import EuroPP
from traffic.data.samples import quickstart


@pytest.fixture
def trajectory_data():
    t = (
        quickstart.query("track==track")
        .assign_id()
        .resample(30)
        .unwrap()
        .eval(max_workers=4, desc="")
    )

    t = t.compute_xy(projection=EuroPP())

    t = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in t
    )
    return t


@pytest.fixture
def navpoints_data():
    navpts = ...
    return navpts
