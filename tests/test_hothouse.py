#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `hothouse` package."""

import pytest

from click.testing import CliRunner

import hothouse
from hothouse import cli
from hothouse.datasets import PLANTS

import numpy as np

@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert 'hothouse.cli.main' in result.output
    help_result = runner.invoke(cli.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output

def test_load_soy():
    fname = PLANTS.fetch('fullSoy_2-12a.ply')
    p = hothouse.plant_model.PlantModel.from_ply(fname)
    assert p.triangles.shape == (3, 3, 8584)

    # Now we will test translation

    r = hothouse.plant_model.PlantModel.from_ply(
            fname, origin = (0.1, 0.2, 0.3))
    assert np.all(p.triangles[:,0,:] + 0.1 == r.triangles[:,0,:])
    assert np.all(p.triangles[:,1,:] + 0.2 == r.triangles[:,1,:])
    assert np.all(p.triangles[:,2,:] + 0.3 == r.triangles[:,2,:])
