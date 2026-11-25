"""Smoke tests for the installed package and CI test tooling."""

from base64 import b64decode, b64encode
from importlib.metadata import entry_points

from hypothesis import given
from hypothesis import strategies as st
from pytest_mock import MockerFixture

import mirage


def test_installed_console_entry_point_calls_main(mocker: MockerFixture) -> None:
    main = mocker.spy(mirage, "main")

    [entry_point] = entry_points(group="console_scripts", name="mirage")
    entry_point.load()()

    main.assert_called_once_with()


@given(st.binary())
def test_base64_round_trip(data: bytes) -> None:
    assert b64decode(b64encode(data)) == data
