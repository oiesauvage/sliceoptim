# -*- coding: utf-8 -*-
"""sliceoptim module for interaction with Slic3r software and generation of samples batches.
"""

__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "apache 2.0"

import logging
from pathlib import Path
from subprocess import CalledProcessError, run
import pandas as pd
import numpy as np
import os

from pandas.core.frame import DataFrame

from sliceoptim import __version__


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sliceoptim.core import Printer, Filament

log = logging.getLogger("sliceoptim" + __name__)


class Sample:
    """Class to define samples to test slicing on. Connects with Slic3r software with cli.
    """

    def __init__(
        self,
        input_file: str,
        is_first_layer: bool,
        printer: "Printer",
        filament: "Filament",
        config_file: Path = None,
        params: dict = None,
    ) -> None:
        """Instantiates a test Sample, holding a stl file and a set of slicing parameters.
           The generated object can produce corresponding gcode used to build test grids.
           Uses Slic3r as slicing engine.

        Args:
            input_file (str): stl file to slice.
            is_first_layer (bool): Declares if the Sample targets first-layer tests.
            printer (Printer): Printer object holding printer's characteristics.
            filament (Filament): Filament object holding filament's characteristics.
            params ([type], optional): Additional parameters to pass to SLic3r. Defaults to None.
            config_file (str, pathlike): file path for a Slic3r .ini config file.
        """
        # params is a dict of slic3r printing options with corresponding values
        self.input_file = input_file
        self.config_file = config_file
        self.printer = printer
        self.filament = filament
        self.is_first_layer = is_first_layer
        self.temp_output_dir = "/tmp/printer_optim"
        self.quality = None
        self.cost = None
        self.__print_time = None
        self.__params = {}
        if is_first_layer:
            self.set_first_layer_presets()
        else:
            self.set_standard_print_presets()
        if (printer is not None) and (filament is not None):
            self.set_main_presets(
                printer=printer, filament=filament
            )  # extract and set first layer temperature params
        if config_file is not None:
            with open(config_file, "r") as conf_file:
                conf = conf_file.read()
                self.set_param(
                    "first-layer-bed-temperature",
                    int(conf.split("first_layer_bed_temperature = ")[1].split("\n")[0]),
                )
                self.set_param(
                    "first-layer-temperature",
                    int(conf.split("first_layer_temperature = ")[1].split("\n")[0]),
                )
        if params is not None:
            for p, v in params.items():
                self.set_param(p, v)
        pass

    @property
    def output_file(self) -> str:
        """Getter for output file.

        Returns:
            str: Returns the path for the test Sample output file.
        """
        if self.get_param("output") == "default":
            filepath = self.input_file.replace(".stl", ".gcode")
        else:
            filepath = self.get_param("output")
        return filepath

    @output_file.setter
    def output_file(self, value: str) -> None:
        """Setter for output file.

        Args:
            value (str): output file path.
        """
        self.set_param("output", value)
        log.debug("Sample output file set to {}".format(value))
        pass

    @property
    def print_time(self) -> float:
        """Getter for Sample print time estimate.

        Returns:
            [float]: Returns estimated print time.
        """
        if self.__print_time is None:
            gcode = self.gcode  # print time is updated at gcode generation
        log.debug("Estimated print time for Sample is {} min".format(self.__print_time))
        return self.__print_time

    @print_time.setter
    def print_time(self, value):
        """[summary]

        Args:
            value ([type]): No print_time value can be set (it is estimated internally).

        Raises:
            AttributeError: Print time can not be set directly.
        """
        raise AttributeError("Print time can not be set (it is computed form gcode).")

    def __estimate_print_time_from_gcode(self, gcode: str) -> float:
        """Estimates print time in minutes from given GCODE.

        Args:
            gcode (str): GCODE whose print time is estimated from.

        Returns:
            float: A print time estimated from simple movements decomposition.
        """

        def parse_line(line: str) -> dict:
            """Parses a line of GCODE towards the resulting coordinates.

            Args:
                line (str): GCODE instruction line.

            Returns:
                dict: New coordinates as a dict.
            """
            move = {"G": None, "X": None, "Y": None, "Z": None, "F": None}
            for key, val in move.items():
                if key in line:
                    d = line.split(key)[1].split(" ")[0]
                    try:
                        if key == "G":
                            move[key] = int(d)
                        else:
                            move[key] = float(d)
                    except ValueError:
                        continue
            return move

        lines = gcode.splitlines()

        minutes = 0
        feedrate = 0
        last_state = {"G": 0, "X": 0, "Y": 0, "Z": 0, "F": 0}
        for l in lines:
            move = parse_line(l)
            distance = 0
            if move["G"] == 1:
                if move["F"] is not None:
                    feedrate = move["F"]
                    last_state["F"] = move["F"]
                for c in ["X", "Y", "Z"]:
                    if move[c] is not None and move[c] != last_state[c]:
                        distance = np.abs(move[c] - last_state[c])
                        last_state[c] = move[c]
            if feedrate != 0 and distance != 0:
                minutes += distance / feedrate
        return minutes

    @property
    def gcode(self) -> str:
        """Getter for generated GCODE.

        Returns:
            str: generated GCODE as a string.
        """
        temp_output_file = self.temp_output_dir + "/sample.gcode"
        if not os.path.isdir(self.temp_output_dir):
            os.makedirs(self.temp_output_dir)
        self.write_gcode(temp_output_file)
        with open(temp_output_file) as file:
            gcode = file.read()
            file.close()
        os.remove(temp_output_file)
        self.__print_time = self.__estimate_print_time_from_gcode(gcode=gcode)
        return gcode

    @gcode.setter
    def gcode(self, value):
        """Setter for GCODE. Raises an error because GCODE is internally computed.

        Args:
            value ([type]): gcode cannot be set.

        Raises:
            AttributeError: GCODE cannot be set (only computed).
        """
        raise AttributeError("GCODE cannot be set (only computed).")

    def erase_start_end_gcode(self) -> None:
        """Erases starting en ending GCODE in Slic3r parameters."""
        self.set_param("start-gcode", "")
        self.set_param("end-gcode", "")
        pass

    def set_param(self, name: str, value) -> None:
        """Set a Slic3r parameter for the current Sample.

        Args:
            name (str): Name of the corresponding Slic3r parameter.
            value ([type]): Desired value for the parameter.
        """
        # remove param if none or default
        if (value is None) or (value == "default"):
            if name in self.__params.keys():
                self.__params.pop(name)
        # convert type if value is a float
        elif isinstance(value, float):
            if ("temperature" in name) or ("speed" in name):
                value = int(value)
            self.__params[name] = value
        else:
            self.__params[name] = value
        # reset computed print times and quality rates when a param is changed
        self.__print_time = None
        self.quality = None
        pass

    def get_param(self, name: str) -> str:
        """Get set value for a SLic3r parameter.

        Args:
            name (str): Name of queried parameter.

        Returns:
            str: Parameter value.
        """
        value = "default"
        if name in self.__params.keys():
            value = self.__params[name]
        return value

    def reset_params(self) -> None:
        """Reset all Slic3r parameters."""
        self.__params = {}
        pass

    def write_gcode(self, output_file=None) -> None:
        """Write GCODE to given / default output file

        Args:
            output_file (str, optional): File path to write test sample GCODE. Defaults to None.

        Raises:
            ValueError: Retrived error from Slic3r call.
        """
        cmd = [
            "slic3r",
        ]
        params = self.__params.copy()
        if self.config_file is not None:
            params["load"] = self.config_file
        if output_file is not None:
            params["output"] = output_file
        for key, val in params.items():
            cmd.append("--" + key)
            if val is not None:
                cmd.append(str(val))
        cmd.append(self.input_file)
        res = run(cmd, capture_output=True)
        try:
            res.check_returncode()
        except CalledProcessError:
            raise ValueError(str(res.stderr))
        log.debug("GCODE generated for Sample")
        pass

    def get_info(self) -> dict:
        """Get Sample information from Slic3r parsing.

        Raises:
            ValueError: Error from Slic3r call.

        Returns:
            dict: Sample information from Slic3r parsing.
                Example of returned info:
                {'filename': 'calicat.stl',
                'size_x': 28.535534,
                'size_y': 28.5,
                'size_z': 35.0,
                'min_x': 0.464466,
                'min_y': 0.0,
                'min_z': 0.0,
                'max_x': 29.0,
                'max_y': 28.5,
                'max_z': 35.0,
                'number_of_facets': 876.0,
                'manifold': 'yes',
                'number_of_parts': 1.0,
                'volume': 12501.378906}
        """
        infos = {}
        cmd = ["slic3r", self.input_file, "--info"]
        res = run(cmd, capture_output=True)
        try:
            res.check_returncode()
        except CalledProcessError:
            raise ValueError(str(res.stderr))
        for inf in res.stdout.decode("utf8").split("\n"):
            if "[" in inf:
                infos["filename"] = inf[1:-1]
            elif " = " in inf:
                inf = inf.split(" = ")
                try:
                    infos[inf[0]] = float(inf[1])
                except ValueError:
                    infos[inf[0]] = inf[1]
        return infos

    def set_first_layer_presets(self) -> None:
        """Set Slic3r parameters for first layer case."""
        self.set_param("top-solid-layers", 0)
        self.set_param("bottom-solid-layers", 1)
        self.set_param("fill-density", 0)
        self.set_param("skirts", 0)
        self.set_param("perimeters", 1)
        log.debug("Set first layer presets for Sample.")
        pass

    def set_standard_print_presets(self) -> None:
        """Set Slic3r presets for a standard sample print."""
        self.set_param("top-solid-layers", 2)
        self.set_param("bottom-solid-layers", 2)
        self.set_param("fill-density", 20)
        self.set_param("skirts", 0)
        self.set_param("perimeters", 2)
        log.debug("Set first standard presets for Sample.")
        pass

    def set_main_presets(self, printer: "Printer", filament: "Filament") -> None:
        """Set Slic3r presets derived from pinter and filament parameters.

        Args:
            printer (Printer): Printer object to derive presets from.
            filament (Filament): Filament object to derive presets from.
        """
        self.set_param(
            "bed-shape",
            "0x0,{x}x0,{x}x{y},0x{y}".format(
                x=printer.bed_size[0], y=printer.bed_size[1]
            ),
        )
        self.set_param("min-print-speed", printer.min_speed)
        self.set_param("max-print-speed", printer.max_speed)
        self.set_param("nozzle-diameter", printer.nozzle_diameter)
        # filament related parameters
        self.set_param("filament-diameter", filament.diameter)
        log.debug("Set main presets for Sample.")
        pass

    def get_param_from_gcode(self, gcode: str, param_name: str) -> float:
        """Get a parameter from a GCODE file sliced with Slic3r.

        Args:
            gcode (str): GCODE file to parse, as a string.
            param_name (str): name of the parameter we look for.

        Raises:
            ValueError: Raises an error if the parameter name is not fond in GCODE.

        Returns:
            float: Returns the parameter value, as a float if possible.
        """
        start_str = "; " + param_name + " = "
        end_str = "\n" if param_name != "filament used" else "mm"
        try:
            val = gcode.split(start_str)[1].split(end_str)[0]
        except IndexError:
            raise ValueError("Parameter name could not be found in GCODE.")
        if "%" in val:
            val = float(val.split("%")[0])
            # special cases for parameters in percent
            if param_name == "first_layer_extrusion_width":
                val = self.get_param_from_gcode(gcode, "first_layer_height") * val / 100
        else:
            val = float(val)
        if val == 0:
            # special cases for default values
            if param_name == "extrusion_width":
                val = 1.25 * self.get_param_from_gcode(gcode, "nozzle_diameter")
        return val


class SampleGrid:
    """Class for generation of test sample batches.
    """

    def __init__(
        self,
        sample_input_file: str,
        is_first_layer: bool,
        printer: "Printer",
        filament: "Filament",
        designs: DataFrame,
        output_path: str,
        spacing=0.0,
        sample_default_params={},
        config_file=None,
    ) -> None:
        """Instantiates a SampleGrid object, holding and generating a batch of test samples.

        Args:
            sample_input_file (str): stl file path for the test sample.
            is_first_layer (bool): True if the test targets a first layer optimization.
            printer (Printer): Printer object
            filament (Filament): Filament object.
            designs (DataFrame): DataFrame of design parameters for samples (each row corresponds to a Sample).
                                 This argument is optional (can be generated by SampleGrid later).
            output_path (str): File path to write generated GCODE.
            spacing (float, optional): Absolute spacing between samples in mm. Defaults to 0.0.
            sample_default_params (dict, optional): Default parameters dict for Samples.
            config_file (str, pathlike): file path for a Slic3r .ini config file.
        """
        self.help_path = Path("src/sliceoptim/data/slic3r_help_file.txt")
        self.sample_input_file = sample_input_file
        self.is_first_layer = is_first_layer
        self.spacing = spacing
        self.printer = printer
        self.filament = filament
        self.output_path = output_path
        self.sample_default_params = sample_default_params
        self.config_file = config_file
        self.__default_sample = self.gen_test_sample()
        self.__default_sample_info = self.__default_sample.get_info()
        self.check_designs_input(designs)
        self.__samples_list = []
        self.__designs = designs
        self.init_gcode = (
            "G28 ; home all axes\n"
            "G1 Z5 F5000 ; lift nozzle\n"
            "G29 A; Activate UBL\n"
            "G29 L1; Load the mesh stored in slot 1 (from G29 S1)\n"
            "G29 J; Probe 3 points\n"
            "G1 X2.0 Y2.0 Z0.0 F5000.0 ; go to edge of print area\n"
            "G92 E0 ; Reset extruder length to zero\n"
            "M190 S40; Set and wait for bed temperature\n"
            "M109 S190; Set and wait for hotend temperature\n"
            "G1 Z0.200 F1000.0 ; Go to Start Z position\n"
            "G1 Y62.0 E9.0 F1000.0 ; intro line\n"
            "G1 Y102.0 E21.5 F1000.0 ; intro line\n"
            "G1 E20; small retract\n"
            "G92 E0.0 ; reset extruder distance position\n"
        )
        self.end_gcode = (
            "M104 S0 ; turn off temperature\n"
            "G28 X0 ; home X axis\n"
            "M84 ; disable motors\n"
        )
        log.info("Created new SampleGrid")

    @property
    def designs(self) -> pd.DataFrame:
        """Getter for Sample designs DataFrame.

        Returns:
            pd.DataFrame: DataFrame of design parameters for samples
                          Each row corresponds to a Sample, and column to a parameter.
        """
        return self.__designs

    @designs.setter
    def designs(self, val: pd.DataFrame):
        """Setter for Sample designs DataFrame.

        Args:
            val (pd.DataFrame): DataFrame of design parameters for samples
                                Each row corresponds to a Sample, and column to a parameter.
        """
        self.check_designs_input(val)
        self.__samples_list = []
        self.__designs = val
        pass

    def check_designs_input(self, designs: pd.DataFrame) -> bool:
        """Utility method to check if designs are properly formatted.

        Args:
            designs (pd.DataFrame): designs DataFrame to test.

        Raises:
            TypeError: Not a DataFrame.
            ValueError: Wrong length case.

        Returns:
            bool: True if designs DataFrame is correct.
        """
        max_s = self.grid_shape()["max_samples_count"]
        if not isinstance(designs, pd.DataFrame):
            raise TypeError('"designs" entry must be a DataFrame')
        elif len(designs) > max_s:
            raise ValueError(
                "length of 'designs' DataFrame can not be greater "
                "than maximum number of samples on heat bed ({})".format(max_s)
            )
        log.debug("SampleGrid designs are OK")
        pass

    @property
    def samples_list(self) -> list:
        """List of Samples in SampleGrid.

        Returns:
            list: All samples as a list.
        """
        if not self.__samples_list:
            samples_list = []
            df_coords = self.gen_sample_coordinates()
            for s in df_coords.iterrows():
                sample_params = {}
                s_index = s[0]
                s_x = s[1]["x"]
                s_y = s[1]["y"]
                coords = str(s_x) + "," + str(s_y)
                sample_params["print-center"] = coords
                sample_params["start-gcode"] = ""
                sample_params["end-gcode"] = ""
                for param_name in self.designs.columns:
                    val = self.designs.iloc[s_index][param_name]
                    sample_params[param_name] = val
                samples_list.append(self.gen_test_sample(sample_params=sample_params))
                if s_index < len(df_coords) - 1:
                    next_x = df_coords.iloc[s_index + 1]["x"]
                    next_y = df_coords.iloc[s_index + 1]["y"]
                    temp = samples_list[-1].get_param("first-layer-temperature")
                    bed_temp = samples_list[-1].get_param("first-layer-bed-temperature")
                    inter_gcode = self.inter_gcode(
                        temperature=temp, bed_temperature=bed_temp, x=next_x, y=next_y
                    )
                    samples_list[-1].set_param("end-gcode", inter_gcode)
            self.__samples_list = samples_list
        return self.__samples_list

    @property
    def quality_list(self) -> list:
        """Returns quality of samples as a list.

        Returns:
            list: quality values of samples.
        """
        return [s.quality for s in self.samples_list]

    @quality_list.setter
    def quality_list(self, val: list) -> None:
        """Sets quality values of samples from a list

        Args:
            val (list): list of samples quality values
        """
        if len(val) != len(self.samples_list):
            raise ValueError(
                "Dimension mismatch between quality values list and samples number."
            )
        for i, s in enumerate(self.samples_list):
            s.quality = val[i]
        pass

    # TODO : implement cost list and interface with dataframe exports/imports
    @property
    def cost_list(self) -> list:
        """Returns quality of samples as a list.

        Returns:
            list: quality values of samples.
        """
        return [s.cost for s in self.samples_list]

    @quality_list.setter
    def cost_list(self, val: list) -> None:
        """Sets quality values of samples from a list

        Args:
            val (list): list of samples quality values
        """
        if len(val) != len(self.samples_list):
            raise ValueError(
                "Dimension mismatch between quality values list and samples number."
            )
        for i, s in enumerate(self.samples_list):
            s.quality = val[i]
        pass

    def gen_test_sample(self, sample_params: dict = {}) -> Sample:
        """Generates and returns a new test Sample (not added to the SampleGrid).

        Args:
            sample_params (dict, optional): Parameters dict for the new sample. Defaults to {}.

        Returns:
            Sample: Generated Sample object.
        """
        params = self.sample_default_params.copy()
        params.update(sample_params)
        sample = Sample(
            input_file=self.sample_input_file,
            is_first_layer=self.is_first_layer,
            printer=self.printer,
            filament=self.filament,
            params=params,
            config_file=self.config_file,
        )
        log.info("New test Sample generated.")
        return sample

    def inter_gcode(self, temperature, bed_temperature, x, y):
        """Generates intermediate GCODE to be inserted between two Samples.

        Args:
            temperature (float): Temperature of the next Sample.
            bed_temperature (float): Bed temperature of the next Sample.
            x (float): X coordinate of the next Sample.
            y (float): Y coordinate of the next Sample.

        Returns:
            str: Generated intermediate GCODE.
        """
        gcode = "; ### Inter-samples GCODE ### \n"
        height = self.__default_sample_info["size_z"] + 0.2
        gcode += "G1 Y{y} Z{z}\n".format(y=y, z=height)
        gcode += "G1 X{}\n".format(x)
        gcode += "G1 Z0.0\n"
        if isinstance(bed_temperature, int):
            gcode += "M190 R{}\n".format(int(bed_temperature))
        if isinstance(temperature, int):
            gcode += "M109 R{}\n".format(int(temperature))
        log.info("Intermediate GCODE generated.")
        gcode += "; ### End of inter-samples GCODE ### \n"
        return gcode

    def grid_shape(self) -> dict:
        """Compute and returns SampleGrid shape data.

        Returns:
            dict: {"max_samples_count": maximum number of Samples,
                   "n_cols": number of columns (X axis),
                   "n_rows": number of rows (Y axis),
                   "size_x": size of X axis in mm,
                   "size_y": size of Y axis in mm}
        """
        max_x = self.printer.bed_size[0] * 0.85
        max_y = self.printer.bed_size[1] * 0.85
        s_info = self.__default_sample_info
        s_x = s_info["size_x"]
        s_y = s_info["size_y"]
        n_cols = np.floor((max_x + self.spacing) / (s_x + self.spacing))
        n_rows = np.floor((max_y + self.spacing) / (s_y + self.spacing))
        n_samples = 0
        if self.is_first_layer:
            n_samples = n_cols * n_rows
        else:
            n_samples = min(n_cols, n_rows)
        size_x = n_cols * (s_x + self.spacing) - self.spacing
        size_y = n_rows * (s_y + self.spacing) - self.spacing
        return {
            "max_samples_count": int(n_samples),
            "n_cols": int(n_cols),
            "n_rows": int(n_rows),
            "size_x": size_x,
            "size_y": size_y,
        }

    def compute_first_sample_coords(self, grid_shape) -> dict:
        """Computes and returns the coordinates for the first Sample.

        Args:
            grid_shape (dict): SampleGrid shape data as returned by "grid_shape" method.

        Returns:
            dict: "x" and "y" coordinates
        """
        sample_info = self.__default_sample_info
        x = 0.5 * (
            self.printer.bed_size[0] - grid_shape["size_x"] + sample_info["size_x"]
        )
        y = self.printer.bed_size[1] - 0.5 * (
            self.printer.bed_size[1] - grid_shape["size_y"] + sample_info["size_y"]
        )
        coords = {"x": x, "y": y}
        log.info(
            "First Sample coordinates on current SampleGrid are X: {x} and Y: {y}.".format(
                **coords
            )
        )
        return coords

    def gen_sample_coordinates(self) -> pd.DataFrame:
        """Generate all sample coordinates and return them as a DataFrame.

        Returns:
            pd.DataFrame: DataFrame of coordinates with columns "x" and "y".
        """
        # occidental reading disposition
        # length of dataframe
        n_samples = len(self.designs)
        # compute grid shape or diag shape : ToDo !
        grid_shape = self.grid_shape()
        # generate coordinates
        sample_info = self.__default_sample_info
        first_coords = self.compute_first_sample_coords(grid_shape)
        x = np.arange(0, grid_shape["n_cols"], 1)
        y = np.arange(0, grid_shape["n_rows"], 1)
        if self.is_first_layer:
            xx, yy = np.meshgrid(x, y)
        else:  # diagonal disposition
            xx = x
            yy = y
        xx = first_coords["x"] + xx * (sample_info["size_x"] + self.spacing)
        yy = first_coords["y"] - yy * (sample_info["size_y"] + self.spacing)
        # add to dataframe
        coords = {"x": xx.flatten()[0:n_samples], "y": yy.flatten()[0:n_samples]}
        df = pd.DataFrame(coords)
        log.info("Samples coordinates generated.")
        return df

    def write_gcode(self, output_path=None):
        """Writes GCODE to the (predefined) output path.

        Args:
            output_path (str, optional): Output file path to write GCODE on. Defaults to None.
                                         If set to None, writes to default output path.
        """
        if output_path is None:
            output_path = self.output_path
        if os.path.exists(output_path):
            os.remove(output_path)
        output = open(output_path, "a+")  # append mode
        output.write(self.init_gcode)
        for s in self.samples_list:
            output.write(s.gcode)
        output.write(self.end_gcode)
        output.close()
        log.info("SampleGrid GCODE written.")
        pass
