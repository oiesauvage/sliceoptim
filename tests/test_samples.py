# -*- coding: utf-8 -*-

import os
import numpy as np
from numpy.testing._private.utils import assert_equal
import pytest
import pandas as pd
import sliceoptim.samples as sp
import sliceoptim.core as core


__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "mit"

test_stl_first_layer = "./assets/first_layer.stl"
test_stl = "./assets/calicat_small.stl"
test_slic3r_config = "./assets/config_slic3r.ini"


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


@pytest.fixture(params=[True, False])
def sample(gcode_tmp_dir, filament, printer, request) -> sp.Sample:
    is_fl = request.param
    if is_fl:
        sample = sp.Sample(
            test_stl_first_layer,
            filament=filament,
            printer=printer,
            is_first_layer=True,
            config_file=test_slic3r_config,
        )
        sample.set_param("output", gcode_tmp_dir / "test_fl.gcode")
    else:
        sample = sp.Sample(
            test_stl, filament=filament, printer=printer, is_first_layer=False
        )
        sample.set_param("output", gcode_tmp_dir / "test.gcode")
    return sample


# tests
class TestSample:
    def test_instantiation(self, gcode_tmp_dir, printer, filament):
        s_first_layer = sp.Sample(
            test_stl_first_layer,
            filament=filament,
            printer=printer,
            is_first_layer=True,
        )
        s_first_layer.set_param("output", gcode_tmp_dir / "test_first_layer.gcode")
        s_not_first_layer = sp.Sample(
            test_stl,
            filament=filament,
            printer=printer,
            is_first_layer=False,
            config_file=test_slic3r_config,
        )
        s_not_first_layer.set_param(
            "output", gcode_tmp_dir / "test_no_first_layer.gcode"
        )
        assert s_first_layer.is_first_layer
        assert not s_not_first_layer.is_first_layer
        assert s_first_layer.get_param("top-solid-layers") == 0
        assert s_not_first_layer.get_param("top-solid-layers") == 2
        assert s_first_layer.get_param("filament-diameter") == 1.75
        # Test if first layer temperatures retrieved from config
        assert s_not_first_layer.get_param("first-layer-bed-temperature") == 65
        assert s_not_first_layer.get_param("first-layer-temperature") == 200

    def test_write_gcode(self, sample):
        sample.write_gcode()
        file = sample.output_file
        assert "Slic3r" in file.read_text()

    def test_gen_gcode(self, sample):
        gcode = sample.gcode
        assert "Slic3r" in gcode
        assert not os.path.exists(sample.temp_output_dir + "/sample.gcode")

    def test_output_file_edition(self, sample: sp.Sample):
        sample.output_file = "testpath"
        assert sample.output_file == "testpath"
        sample.output_file = None
        output_file = str(sample.output_file)
        if sample.is_first_layer:
            assert output_file == test_stl_first_layer.replace(".stl", ".gcode")
        else:
            assert output_file == test_stl.replace(".stl", ".gcode")

    def test_set_parameters(self, sample):
        sample.set_param("top-solid-layers", 0)
        sample.set_param("top-infill-pattern", "concentric")
        sample.set_param("temperature", 200.15)
        sample.set_param("layer-height", 0.1)
        sample.set_param("first-layer-speed", 60.15)
        sample.set_param("end-gcode", "pouet")
        assert "top_solid_layers = 0" in sample.gcode
        assert "top_infill_pattern = concentric" in sample.gcode
        assert "temperature = 200" in sample.gcode
        assert sample.get_param("first-layer-speed") == 60
        assert sample.get_param("end-gcode") == "pouet"

    def test_params_properly_set_at_init(self, gcode_tmp_dir, printer, filament):
        params = {"temperature": 200.15, "first-layer-speed": 60.15}
        sample = sp.Sample(
            test_stl_first_layer,
            filament=filament,
            printer=printer,
            is_first_layer=True,
            params=params,
        )
        sample.get_param("temperature") == 200
        sample.get_param("first-layer-speed") == 60.15

    def test_remove_parameter(self, sample):
        sample.set_param("top-solid-layers", 0)
        sample.set_param("top-solid-layers", None)
        assert sample.get_param("top-solid-layers") == "default"
        sample.set_param("top-solid-layers", 15)
        sample.set_param("top-solid-layers", "default")
        assert sample.get_param("top-solid-layers") == "default"

    def test_reset_parameters(self, sample):
        sample.set_param("top_solid_layers", 0)
        sample.reset_params()
        assert sample.get_param("top_solid_layers") == "default"

    def test_get_parameters(self, sample):
        assert sample.get_param("top_infill_pattern") == "default"
        sample.set_param("top_solid_layers", 0)
        sample.set_param("top_infill_pattern", "concentric")
        assert sample.get_param("top_solid_layers") == 0
        assert sample.get_param("top_infill_pattern") == "concentric"

    def test_gen_failed(self, sample):
        with pytest.raises(ValueError):
            sample.set_param("top_solid_layers", -1)
            sample.write_gcode()

    def test_get_stl_info(self, sample: sp.Sample):
        info = sample.get_info()
        if sample.is_first_layer:
            assert info["filename"] in test_stl_first_layer
        else:
            assert info["filename"] in test_stl
        assert type(info["size_x"]) == float
        assert type(info["manifold"]) == str

    def test_erase_start_end_gcode(self, sample):
        sample.set_param("start-gcode", "G28")
        sample.set_param("end-gcode", "G28")
        sample.erase_start_end_gcode()
        assert sample.get_param("start-gcode") == ""
        assert sample.get_param("end-gcode") == ""

    def test_set_first_layer_presets(self, sample: sp.Sample):
        sample.set_first_layer_presets()
        assert sample.get_param("top-solid-layers") == 0
        assert sample.get_param("bottom-solid-layers") == 1
        assert sample.get_param("fill-density") == 0
        assert sample.get_param("skirts") == 0
        assert sample.get_param("perimeters") == 1

    def test_set_standard_layer_presets(self, sample: sp.Sample):
        sample.set_standard_print_presets()
        assert sample.get_param("top-solid-layers") == 2
        assert sample.get_param("bottom-solid-layers") == 2
        assert sample.get_param("fill-density") == 20
        assert sample.get_param("skirts") == 0
        assert sample.get_param("perimeters") == 2

    def test_set_main_presets(
        self, sample: sp.Sample, printer: core.Printer, filament: core.Filament
    ):
        sample.set_main_presets(printer=printer, filament=filament)
        assert sample.get_param("bed-shape") == "0x0,{x}x0,{x}x{y},0x{y}".format(
            x=printer.bed_size[0], y=printer.bed_size[1]
        )
        assert sample.get_param("min-print-speed") == printer.min_speed
        assert sample.get_param("max-print-speed") == printer.max_speed
        assert sample.get_param("nozzle-diameter") == printer.nozzle_diameter
        # filament related parameters
        assert sample.get_param("filament-diameter") == filament.diameter

    def test_get_param_from_gcode(self, sample):
        gcode = sample.gcode
        assert isinstance(sample.get_param_from_gcode(gcode, "filament used"), float)
        assert isinstance(
            sample.get_param_from_gcode(gcode, "first_layer_height"), float
        )

    def test_print_time_accessor(self, sample: sp.Sample):
        sample.set_param("extrusion-width", 0.4)
        if sample.is_first_layer:
            assert sample.print_time == pytest.approx(0.431, 0.1)
        else:
            assert sample.print_time == pytest.approx(7.14, 0.1)
        with pytest.raises(AttributeError):
            sample.print_time = 100.0

    def test_quality_reset_when_params_changed(self, sample):
        sample.quality = 10.0
        sample.set_param("extrusion-width", 0.6)
        assert sample.quality is None


@pytest.fixture(params=["first_layer", "not_first_layer"])
def sample_grid(printer, filament, gcode_tmp_dir, request) -> sp.SampleGrid:
    if request.param == "first_layer":
        stl = test_stl_first_layer
        is_fl = True
        s = 1
        n = 48
        conf = {}
    elif request.param == "not_first_layer":
        stl = test_stl
        is_fl = False
        s = 30
        n = 3
        conf = {}
    d = gcode_tmp_dir
    bed_temperature = 50 + 30 * np.random.random(n)
    temperature = 180 + 40 * np.random.random(n)
    designs = pd.DataFrame(
        {"temperature": temperature, "bed-temperature": bed_temperature}
    )
    grid = sp.SampleGrid(
        designs=designs,
        sample_input_file=stl,
        is_first_layer=is_fl,
        printer=printer,
        filament=filament,
        output_path=d / "test_grid.gcode",
        spacing=s,
        sample_default_params=conf,
        config_file=test_slic3r_config,
    )
    return grid


class TestSampleGrid:
    def test_gen_test_sample(self, sample_grid):
        s1 = sample_grid.gen_test_sample()
        s2 = sample_grid.gen_test_sample()
        s1.set_param("extrusion-width", 0.6)
        assert s1.get_param("extrusion-width") == 0.6
        assert s1.get_param("extrusion-width") != s2.get_param("extrusion-width")

    def test_compute_grid_shape(self, sample_grid: sp.SampleGrid):
        shape = sample_grid.grid_shape()
        if sample_grid.is_first_layer:
            assert shape["n_rows"] == 7
            assert shape["n_cols"] == 7
            assert shape["size_x"] == 7 * 25 + 6
            assert shape["size_y"] == 7 * 25 + 6
        else:
            assert shape["n_rows"] == 4
            assert shape["n_cols"] == 4
            assert shape["size_x"] == pytest.approx(147, 0.1)
            assert shape["size_y"] == pytest.approx(147, 0.1)

    def test_compute_first_sample_coords(self, sample_grid: sp.SampleGrid):
        grid_shape = sample_grid.grid_shape()
        coords = sample_grid.compute_first_sample_coords(grid_shape)
        if sample_grid.is_first_layer:
            assert coords["x"] == pytest.approx(32, 0.1)
            assert coords["y"] == pytest.approx(220 - 32, 0.1)
        else:
            assert coords["x"] == pytest.approx(43.6, 0.1)
            assert coords["y"] == pytest.approx(220 - 43.6, 0.1)

    def test_write_gcode(self, sample_grid):
        output_file = str(sample_grid.output_path)
        output_file = output_file.replace("test.gcode", "fused_testfile.gcode")
        sample_grid.designs = sample_grid.designs[0:3]
        sample_grid.write_gcode(output_file)
        assert os.path.exists(output_file)
        file = open(output_file)
        gcode = file.read()
        for param in sample_grid.designs.columns:
            for val in sample_grid.designs[param].values:
                assert param.replace("-", "_") + " = " + str(int(val)) in gcode

    def test_inter_gcode_written_in_output(self, sample_grid):
        sample_grid.designs = sample_grid.designs[0:3]
        output_file = str(sample_grid.output_path)
        output_file = output_file.replace("test.gcode", "fused_testfile.gcode")
        sample_grid.write_gcode(output_file)
        file = open(output_file)
        gcode = file.read()
        assert sample_grid.samples_list[-2].get_param("end-gcode") in gcode

    def test_get_max_samples(self, sample_grid: sp.SampleGrid):
        shape = sample_grid.grid_shape()
        if sample_grid.is_first_layer:
            assert shape["max_samples_count"] == 7 * 7
        else:
            assert shape["max_samples_count"] == 4

    def test_limit_samples_number(self, sample_grid: sp.SampleGrid):
        designs = sample_grid.designs
        df = designs.append(designs)
        with pytest.raises(ValueError):
            sample_grid.designs = df

    def test_set_quality(self, sample_grid: sp.SampleGrid):
        quality = [i for i in range(len(sample_grid.designs))]
        sample_grid.quality_list = quality
        assert [s.quality for s in sample_grid.samples_list] == quality

    def test_get_quality(self, sample_grid: sp.SampleGrid):
        quality = [i for i in range(len(sample_grid.designs))]
        sample_grid.quality_list = quality
        assert sample_grid.quality_list == quality
