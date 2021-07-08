# -*- coding: utf-8 -*-

from os import get_exec_path
from _pytest.monkeypatch import V
from copy import copy
import pytest
import sliceoptim.io as io
import sliceoptim.core as core
from skopt import Space
from skopt.space import Real, Integer

__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "mit"

test_stl = "./assets/calicat_small.stl"


@pytest.fixture()
def db_tmp_dir(tmp_path):
    d = tmp_path / "database"
    d.mkdir()
    return d


@pytest.fixture
def database(db_tmp_dir) -> io.Database:
    db = io.Database(folder_path=db_tmp_dir)
    return db


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


@pytest.fixture()
def experiment(
    printer, filament, gcode_tmp_dir, params_space, request
) -> core.Experiment:
    out = gcode_tmp_dir / "test.gcode"
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


class TestDatabase:
    def test_init_database(self, db_tmp_dir):
        db = io.Database(folder_path=db_tmp_dir)
        assert isinstance(db, io.Database)

    def test_manage_filament(self, database: io.Database, filament):
        database.add_filament(filament)
        with pytest.raises(ValueError):
            database.add_filament(filament)
        filament_2 = copy(filament)
        filament_2.name = filament.name + "_two"
        database.add_filament(filament_2)
        assert database.get_filament_names() == [filament.name, filament_2.name]
        filament_from_db = database.get_filament(filament.name)
        assert filament.__dict__ == filament_from_db.__dict__
        database.delete_filament(filament.name)
        with pytest.raises(ValueError):
            database.get_filament(filament.name)

    def test_manage_printer(self, database: io.Database, printer):
        database.add_printer(printer)
        with pytest.raises(ValueError):
            database.add_printer(printer)
        printer_2 = copy(printer)
        printer_2.name = printer.name + "_two"
        database.add_printer(printer_2)
        assert database.get_printer_names() == [printer.name, printer_2.name]
        printer_from_db = database.get_printer(printer.name)
        assert printer.__dict__ == printer_from_db.__dict__
        database.delete_printer(printer.name)
        with pytest.raises(ValueError):
            database.get_printer(printer.name)

    def test_manage_experiment(
        self, database: io.Database, experiment: core.Experiment
    ):
        database.add_experiment(experiment=experiment)
        with pytest.raises(ValueError):
            database.add_experiment(experiment=experiment)
        experiment_2 = copy(experiment)
        experiment_2.name = experiment.name + "_two"
        database.add_experiment(experiment_2)
        assert database.get_experiment_names() == [experiment.name, experiment_2.name]
        exp_from_db = database.get_experiment(experiment.name)
        assert isinstance(exp_from_db, core.Experiment)
        assert experiment.name == exp_from_db.name
        experiment.create_new_sample_grid(2)
        database.update_experiment(experiment=experiment)
        assert len(database.get_experiment(experiment.name).to_dataframe()) == 2
        database.delete_experiment(experiment.name)
        with pytest.raises(ValueError):
            database.get_experiment(experiment.name)

