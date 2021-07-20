# -*- coding: utf-8 -*-

import os
from attr import validate
from pandas import read_csv
from pandas.core.frame import DataFrame, Series
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import pytest
import skopt
import sliceoptim.core as core
from skopt import Space, Optimizer
from skopt.space import Real, Integer
from copy import copy

from sliceoptim.samples import Sample, SampleGrid


__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "mit"

test_stl_first_layer = "./assets/first_layer.stl"
test_stl = "./assets/calicat_small.stl"

# fixtures
@pytest.fixture
def printer() -> core.Printer:
    printer = core.Printer(
        name="test_printer",
        bed_size=[220, 220],
        nozzle_diameter=0.4,
        max_speed=120,
        min_speed=5,
    )
    return printer


@pytest.fixture
def filament() -> core.Filament:
    filament = core.Filament(
        name="test_filament",
        material="pla",
        extrusion_temp_range=[180, 250],
        bed_temp_range=[25, 80],
        diameter=1.75,
    )
    return filament


@pytest.fixture()
def gcode_tmp_dir(tmp_path):
    d = tmp_path / "gcode"
    d.mkdir()
    return d


@pytest.fixture
def params_space(printer) -> Space:
    params = []
    params.append(Integer(name="first-layer-temperature", low=190, high=220))
    params.append(Integer(name="first-layer-bed-temperature", low=40, high=55))
    params.append(
        Integer(
            name="first-layer-speed",
            low=printer.min_speed,
            high=printer.max_speed * 0.5,
        )
    )
    params.append(
        Real(
            name="first-layer-height",
            low=printer.nozzle_diameter * 0.75,
            high=printer.nozzle_diameter,
        )
    )
    params.append(
        Real(
            name="first-layer-extrusion-width",
            low=printer.nozzle_diameter * 0.9,
            high=printer.nozzle_diameter * 2.5,
        )
    )
    params.append(Real(name="extrusion-multiplier", low=0.8, high=1.2))
    params.append(
        Real(
            name="layer-height",
            low=printer.nozzle_diameter * 0.2,
            high=printer.nozzle_diameter,
        )
    )
    space = Space(params)
    return space


@pytest.fixture(params=[True, False])
def experiment(
    printer, filament, gcode_tmp_dir, params_space, request
) -> core.Experiment:
    out = gcode_tmp_dir / "test.gcode"
    if request:
        stl = test_stl_first_layer
        is_fl = True
        s = 1
    else:
        stl = test_stl
        is_fl = False
        s = 30
    exp = core.Experiment(
        "test",
        sample_file=stl,
        is_first_layer=is_fl,
        spacing=s,
        printer=printer,
        filament=filament,
        params_space=params_space,
        output_file=out,
    )
    return exp


class TestParametersSpace:
    def test_init_params_space(self):
        p_space = core.ParametersSpace()
        assert isinstance(p_space, core.ParametersSpace)

    def test_add_param(self):
        p_space = core.ParametersSpace()
        p_space.add_param("temperature", low=180, high=250.5)
        assert "temperature" in p_space.dimension_names
        assert isinstance(p_space.dimensions[-1], Integer)
        p_space.add_param("extrusion-multiplier", low=0.7, high=1.2)
        assert "extrusion-multiplier" in p_space.dimension_names
        assert isinstance(p_space.dimensions[-1], Real)
        with pytest.raises(KeyError):
            p_space.add_param("foo", low=0, high=10)
        with pytest.raises(ValueError):
            p_space.add_param("bed-temperature", low=80, high=10)

    def test_delete_param(self):
        p_space = core.ParametersSpace()
        p_space.add_param("temperature", low=180, high=250.5)
        p_space.delete_param("temperature")
        assert "temperature" not in p_space.dimension_names
        assert len(p_space.dimensions) == 0
        with pytest.raises(KeyError):
            p_space.delete_param("temperature")


class TestPrinter:
    def test_init_printer(self):
        printer = core.Printer(
            name="test",
            bed_size=[220, 200],
            nozzle_diameter=0.4,
            max_speed=200,
            min_speed=5,
        )
        assert isinstance(printer, core.Printer)


class TestFilament:
    def test_init_filament(self):
        filament = core.Filament(
            name="test",
            material="pla",
            extrusion_temp_range=[180, 250],
            bed_temp_range=[25, 80],
            diameter=1.75,
        )
        assert isinstance(filament, core.Filament)


class TestExperiment:
    def test_init_experiment(self, printer, filament, gcode_tmp_dir, params_space):
        out = gcode_tmp_dir / "test.gcode"
        exp = core.Experiment(
            "test",
            sample_file=test_stl,
            is_first_layer=False,
            spacing=30.0,
            printer=printer,
            filament=filament,
            params_space=params_space,
            output_file=out,
        )
        assert isinstance(exp, core.Experiment)
        assert isinstance(exp.max_samples_count, int)

    def test_generate_designs(self, experiment: core.Experiment):
        n_samples = 5
        with pytest.raises(core.ExperimentError):
            designs = experiment.generate_designs(n_samples)
        experiment.optimizer = Optimizer(
            experiment.params_space,
            "GP",
            acq_func="gp_hedge",
            acq_optimizer="auto",
            initial_point_generator="lhs",
            n_initial_points=n_samples + 1,
        )
        designs = experiment.generate_designs(n_samples=5)
        assert isinstance(designs, DataFrame)
        assert len(designs) == n_samples

    def test_create_new_sample_grid(self, experiment: core.Experiment):
        assert not experiment.sample_grid_list
        experiment.create_new_sample_grid(3)
        assert isinstance(experiment.sample_grid_list[0], SampleGrid)
        experiment.create_new_sample_grid(2)
        assert isinstance(experiment.sample_grid_list[1], SampleGrid)
        assert len(experiment.sample_grid_list[1].designs) == 2
        with pytest.raises(ValueError):
            experiment.create_new_sample_grid(experiment.max_samples_count + 1)

    def test_write_gcode_last_sample_grid(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        experiment.write_gcode_for_last_sample_grid()

    def test_get_samples_list(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        experiment.create_new_sample_grid(2)
        samples_list = experiment.get_samples_list()
        assert len(samples_list) == 5
        for s in samples_list:
            assert isinstance(s, Sample)

    def test_get_samples_results_df(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        experiment.write_gcode_for_last_sample_grid()
        for i, s in enumerate(experiment.sample_grid_list[-1].samples_list):
            s.quality = i
        res_df = experiment.get_samples_results_df()
        assert list(res_df.columns) == ["print_time", "quality", "cost"]
        assert res_df["quality"].tolist() == [0, 1, 2]
        experiment.create_new_sample_grid(2)
        assert len(experiment.get_samples_results_df()) == 5

    def test_compute_costs_from_results_df(self, experiment: core.Experiment):
        df = DataFrame(
            data={
                "print_time": [20, 22, 24, 22.5, 24],
                "quality": [5, 7.0, 9, 7.2, 9.5],
                "cost": [None, None, None, None, None],
            }
        )
        df_costs = experiment.compute_costs_from_results_df(df)
        df_costs.drop(columns="qr", inplace=True)
        df_new = DataFrame(
            {
                "print_time": [21, 25, 24, 20, 19],
                "quality": [8, 7.0, 3, 5, 9],
                "cost": [None, None, None, None, None],
            }
        )
        df = df_costs.append(df_new, ignore_index=True)
        costs = experiment.compute_costs_from_results_df(df)["cost"]
        expected_costs = Series(
            name="cost", data=[10.0, 5.0, 2.5, 7.5, 0.0, 3.9, 7.8, 11.7, 9.1, 1.3],
        )
        assert_series_equal(costs, expected_costs, rtol=0.1)

    def test_compute_and_update_samples_costs(self, experiment: core.Experiment):
        with pytest.raises(core.ExperimentError):
            experiment.compute_and_update_samples_costs()
        with pytest.raises(core.ExperimentError):
            experiment.create_new_sample_grid(3)
            experiment.sample_grid_list[0].samples_list[1].quality = 1
            experiment.compute_and_update_samples_costs()
        samples = experiment.get_samples_list()
        for i, s in enumerate(samples):
            s.quality = i
        experiment.compute_and_update_samples_costs()
        experiment.create_new_sample_grid(3)
        samples = experiment.get_samples_list()
        for i, s in enumerate(samples):
            s.quality = i
        experiment.compute_and_update_samples_costs()
        for s in experiment.get_samples_list():
            assert isinstance(s.cost, float)

    def test_get_designs(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        experiment.create_new_sample_grid(2)
        designs = experiment.get_designs()
        assert isinstance(designs, DataFrame)
        assert len(designs) == 5
        assert designs.index.to_list() == [i for i in range(len(designs))]

    def test_register_results_to_optimizer(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        experiment.create_new_sample_grid(2)
        samples = experiment.get_samples_list()
        for i, s in enumerate(samples):
            s.quality = i
        with pytest.raises(core.ExperimentError):
            experiment.register_costs_to_optimizer()
        experiment.compute_and_update_samples_costs()
        experiment.register_costs_to_optimizer()
        assert len(experiment.optimizer.yi) == 5

    def test_estim_best_config(self, experiment: core.Experiment):
        experiment.create_new_sample_grid(3)
        with pytest.raises(core.ExperimentError):
            experiment.estim_best_config()
        experiment.create_new_sample_grid(2)
        samples = experiment.get_samples_list()
        for i, s in enumerate(samples):
            s.quality = i
        experiment.compute_and_update_samples_costs()
        experiment.register_costs_to_optimizer()
        conf, err, fmin = experiment.estim_best_config()
        assert isinstance(conf, dict)
        assert len(conf) == len([p for p in experiment.params_space])

    def test_export_csv(self, experiment: core.Experiment, tmp_path):
        out_file_path = os.path.join(tmp_path, "results_" + experiment.name + ".csv")
        with pytest.raises(core.ExperimentError):
            experiment.export_csv(out_file_path)
        experiment.create_new_sample_grid(3)
        experiment.export_csv(out_file_path)
        df = read_csv(out_file_path, index_col=0)
        assert isinstance(df, DataFrame)
        param_names = experiment.params_space.dimension_names
        res_names = experiment.get_samples_results_df().columns.to_list()
        out_names = param_names + res_names + ["sample_grid_id"]
        for name in out_names:
            assert name in df.columns

    def test_import_csv(self, experiment: core.Experiment, tmp_path):
        out_file_path = os.path.join(tmp_path, "results_" + experiment.name + ".csv")
        expe_init = copy(experiment)
        experiment.create_new_sample_grid(3)
        experiment.create_new_sample_grid(2)
        samples = experiment.get_samples_list()
        for i, s in enumerate(samples):
            s.quality = i
        experiment.export_csv(out_file_path)
        expe_init.import_csv(out_file_path, overwrite_samples=True)
        assert_frame_equal(
            left=expe_init.get_designs(), right=experiment.get_designs(), rtol=0.25
        )
        assert_frame_equal(
            left=expe_init.get_samples_results_df(),
            right=experiment.get_samples_results_df(),
            rtol=0.1,
        )

    def test_init_end_gcode(self, experiment: core.Experiment):
        experiment.init_gcode = "test_init"
        experiment.end_gcode = "test_end"
        assert experiment.init_gcode == "test_init"
        assert experiment.end_gcode == "test_end"

