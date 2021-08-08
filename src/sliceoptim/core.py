# -*- coding: utf-8 -*-
"""Core module of sliceoptim
Contains main classes for definition of filaments, printers and experiments.
"""

__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "apache 2.0"

from pandas.core.frame import DataFrame
import skopt
import pandas as pd
from sliceoptim.samples import Sample, SampleGrid
from typing import List, Tuple
from copy import copy
import numpy as np
import yaml
import logging
from pathlib import Path


from sliceoptim import __version__


log = logging.getLogger("sliceoptim" + __name__)


class ExperimentError(Exception):
    pass


class ParametersSpace(skopt.Space):
    """Class for parametric space expored by Experiment."""

    def __init__(self) -> None:
        """Creates a new ParametersSpace."""
        super().__init__([])
        self.__params_spec_file = (
            Path(__file__).parent / "data" / "implemented_params.yml"
        )
        with open(self.__params_spec_file, "r") as file:
            params_spec = yaml.safe_load(file)
        self.params_spec = params_spec
        return

    def add_param(self, name: str, low: float, high: float):
        """Adds a new parameter.

        Args:
            name (str): Parameter name.
            low (float): Lower bound for parameter values.
            high (float): Upper bound for parameter values.

        Raises:
            KeyError: Error if parameter not supported.
            ValueError: Error if inverted bounds.
        """
        if name not in self.params_spec.keys():
            raise KeyError(
                "The parameter {} doesn't exist or is not yet implemented.".format(name)
            )
        if low >= high:
            raise ValueError("low anf high bounds are inverted or equal.")
        param = self.params_spec[name]
        if param["type"] == "float":
            var = skopt.space.Real(name=name, low=low, high=high)
        elif param["type"] == "int":
            var = skopt.space.Integer(name=name, low=int(low), high=int(high))
        self.dimensions.append(var)
        pass

    def delete_param(self, name: str):
        """Delete a parameter.

        Args:
            name (str): Parameter name.

        Raises:
            KeyError: Error if parameter not existing.
        """
        if name not in self.dimension_names:
            raise KeyError(
                "The parameter {} is not in current parameters space.".format(name)
            )
        self.dimensions.pop(self.dimension_names.index(name))
        pass


class Printer:
    """Printer class to handle printer parameters."""

    def __init__(
        self,
        name: str,
        bed_size: List[float],
        nozzle_diameter: float,
        max_speed: float,
        min_speed: float,
    ):
        """Creates a new Printer object.

        Args:
            name (str): Printer name
            bed_size (List(float)): [X size of bed, Y size of bed]
            nozzle_diameter (float): Nozzle diameter in mm.
            max_speed (float): maximum moves speed in mm/s.
            min_speed (float): minimum moves speed in mm/s.
        """
        self.name = name
        self.bed_size = bed_size
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.nozzle_diameter = nozzle_diameter
        log.info("Created new Printer: {}".format(name))
        pass


class Filament:
    """Filament class to handle filament parameters."""

    def __init__(
        self,
        name: str,
        material: str,
        extrusion_temp_range: List[float],
        bed_temp_range: List[float],
        diameter: float,
    ):
        """Creates a new Filament object

        Args:
            name (str): Name of the Filament.
            kind (str): Filament kind (PLA, PET-G, ABS, TPU...)
            extrusion_temp_range (List[float]): Temperature range in Celsius.
            bed_temp_range (List[float]): Bed temperature range in Celsius.
            diameter (float): Filament diameter in mm.
        """
        self.name = name
        self.material = material
        self.extrusion_temp_range = extrusion_temp_range
        self.bed_temp_range = bed_temp_range
        self.diameter = diameter
        log.info("Created new Filament: {}".format(name))
        pass


class Experiment:
    """Class to handle experiments on slicing parameters, generate test batches and compute optimal results."""

    def __init__(
        self,
        name: str,
        sample_file: str,
        is_first_layer: bool,
        spacing: float,
        printer: Printer,
        filament: Filament,
        params_space: ParametersSpace,
        output_file: str,
        sample_default_params={},
        config_file=None,
    ):
        """Creates a new Experiment instance to handle successive tests.

        Args:
            name (str): Name of the new experiment.
            sample_file (str): stl sample file used as a basis for generated test samples.
            is_first_layer (bool): Selects if the experiment is for a first-layer case.
            spacing (float): Absolute spacing between test samples (in mm).
            printer (Printer): Printer instance providing printer's caracteristics.
            filament (Filament): Filament instance providing filament's carateristics.
            params_space (skopt.Space): skopt.Space instance describing the parametric space to optimize.
            output_file (str): Path for generated gcode files.
            config_file (str, pathlike): file path for a Slic3r .ini config file.
        """
        self.name = name
        self.params_space = params_space
        self.is_first_layer = is_first_layer
        self.sample_file = sample_file
        self.config_file = config_file
        self.printer = printer
        self.filament = filament
        self.__dummy_sample_grid = SampleGrid(
            sample_input_file=sample_file,
            designs=pd.DataFrame(),
            is_first_layer=is_first_layer,
            printer=printer,
            filament=filament,
            spacing=spacing,
            output_path=output_file,
            sample_default_params=sample_default_params,
            config_file=config_file,
        )
        self.max_samples_count = self.__dummy_sample_grid.grid_shape()[
            "max_samples_count"
        ]
        self.spacing = spacing
        self.sample_grid_list = []
        self.optimizer = None
        log.info("Created new Experiment: {}".format(name))
        pass

    @property
    def init_gcode(self) -> str:
        return self.__dummy_sample_grid.init_gcode

    @init_gcode.setter
    def init_gcode(self, value):
        self.__dummy_sample_grid.init_gcode = value
        pass

    @property
    def end_gcode(self) -> str:
        return self.__dummy_sample_grid.end_gcode

    @end_gcode.setter
    def end_gcode(self, value):
        self.__dummy_sample_grid.end_gcode = value
        pass

    def create_new_sample_grid(self, n_samples: int) -> SampleGrid:
        """Creates a new sample grid (test samples to print)
           given a required number of samples (not greater than
           the maximum bed capacity).

        Args:
            n_samples (int): number of samples for the new samples grid.

        Raises:
            ValueError: The number of samples can't be above a limit (computed from 3d printer bed space).

        Returns:
            SampleGrid: The new generated sample grid.
        """
        if n_samples > self.max_samples_count:
            raise ValueError(
                "length of 'designs' pd.DataFrame can not be greater "
                "than maximum number of samples on heat bed ({})".format(
                    self.max_samples_count
                )
            )
        sample_grid = copy(self.__dummy_sample_grid)
        if not self.sample_grid_list:
            # first sample grid case
            # init optimizer and create first designs
            self.optimizer = skopt.Optimizer(
                self.params_space,
                "GP",
                acq_func="gp_hedge",
                acq_optimizer="auto",
                initial_point_generator="lhs",
                n_initial_points=n_samples + 1,
            )
        designs = self.generate_designs(n_samples)
        sample_grid.designs = designs
        self.sample_grid_list.append(sample_grid)
        return sample_grid

    def generate_designs(self, n_samples: int) -> pd.DataFrame:
        """Generate a DataFrame of new design parameters.

        Args:
            n_samples (int): Number of designs to generate.

        Raises:
            ExperimentError: Can't generate more designs than the printer's bed can hold.

        Returns:
            pd.DataFrame: DataFrame of designs parameters' values.
        """
        if self.optimizer is None:
            raise ExperimentError(
                "Can't generate new designs if an optimizer is not present! "
                "Create a new sample grid to generate it automatically."
            )
        points = self.optimizer.ask(n_samples)
        designs = pd.DataFrame(points, columns=[p.name for p in self.params_space])
        if self.is_first_layer:
            if "first-layer-temperature" in designs.columns:
                designs["temperature"] = designs["first-layer-temperature"]
            if "first-layer-bed-temperature" in designs.columns:
                designs["bed-temperature"] = designs["first-layer-bed-temperature"]
        if "first-layer-bed-temperature" in designs.columns:
            designs = designs.sort_values("first-layer-bed-temperature").set_index(
                designs.index
            )
        log.info("New designs generated!")
        return designs

    def write_gcode_for_last_sample_grid(self):
        """Write to output file the gcode for the last test sample grid."""
        last_sample_grid = self.sample_grid_list[-1]
        last_sample_grid.write_gcode()
        pass

    def get_samples_list(self) -> List[Sample]:
        """Return all test samples generated so far as a list.

        Returns:
            List[Sample]: list of generated test samples.
        """
        samples = []
        for sg in self.sample_grid_list:
            for s in sg.samples_list:
                samples.append(s)
        return samples

    def get_samples_results_df(self) -> pd.DataFrame:
        """Return a DataFrame of results for each generated test sample
           (print time, quality, and "cost" computed by the objective function).

        Returns:
            pd.DataFrame: DataFrame of all samples results.
        """
        samples = self.get_samples_list()
        print_times = [s.print_time for s in samples]
        qualities = [s.quality for s in samples]
        costs = [s.cost for s in samples]
        df = pd.DataFrame(
            {"print_time": print_times, "quality": qualities, "cost": costs}
        )
        return df

    def compute_costs_from_results_df(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Compute "costs" to be minimized by the optimizer.
           Costs are computed to prioritize quality over speed
           (ie, reduce speed only if print quality impact is limited).
           Fist costs are computed using a linear interpolation technique,
           and next ones using a spline interpolation.

        Args:
            results_df (pd.DataFrame): DataFrame containing all samples results.

        Returns:
            pd.DataFrame: Input DataFrame populated with samples' cost results.
        """
        df = results_df.copy()
        df["qr"] = df["quality"].apply(lambda x: np.around(x * 2) / 2)
        df.sort_values(by=["qr", "print_time"], ascending=[True, False], inplace=True)
        if df["cost"].isnull().all():
            # case of first results
            df["cost"] = np.linspace(start=10, stop=0, num=len(df))
        else:
            df["cost"] = (
                df["cost"]
                .reset_index()
                .interpolate(method="spline", order=1, limit_direction="both")
                .set_index("index")
                .sort_index()
            )
        df.sort_index(inplace=True)
        log.info("Optimization costs computed for last generated Samples.")
        return df

    def compute_and_update_samples_costs(self):
        """Compute samples' cost from speed and quality results and update costs in samples objects.

        Raises:
            ExperimentError: At least one grid of test samples must be generated.
            ExperimentError: All samples must have a quality result registered.
        """
        if not self.sample_grid_list:
            raise ExperimentError(
                "No sample grid registered."
                "Please create one first with create_new_sample_grid() method."
            )
        df_res = self.get_samples_results_df()
        if df_res["quality"].isnull().any():
            raise ExperimentError(
                "Not all quality results of samples has been registered."
            )
        cost = self.compute_costs_from_results_df(df_res)["cost"]
        samples = self.get_samples_list()
        for i, s in enumerate(samples):
            s.cost = cost.iloc[i]
        log.info("Samples optimization costs updated.")
        pass

    def get_designs(self) -> DataFrame:
        """Returns generated test sample designs as a DataFrame.

        Returns:
            DataFrame: Generated test sample designs as a DataFrame.
        """
        designs = DataFrame()
        for sg in self.sample_grid_list:
            designs = designs.append(sg.designs)
        designs.reset_index(drop=True, inplace=True)
        return designs

    def register_costs_to_optimizer(self):
        """Register new computed samples costs to optimizer.

        Raises:
            ExperimentError: All samples costs must be computed before registering them to the optimizer.
        """
        samples = self.get_samples_list()
        samples = samples[len(self.optimizer.yi) :]
        costs = [s.cost for s in samples]
        if not all([isinstance(c, float) for c in costs]):
            raise ExperimentError("All samples costs are not computed.")
        param_names = [p.name for p in self.params_space]
        designs = (
            self.get_designs()
            .iloc[len(self.optimizer.yi) :][param_names]
            .reset_index(drop=True)
        )
        for d in designs.iterrows():
            self.optimizer.tell(list(d[1]), costs[d[0]])
        log.info("Samples optimization costs registered to optimizer.")
        pass

    def estim_best_config(self) -> Tuple[dict, np.array]:
        """Estimate best values for the set of optimized parameters, given actual test samples results.

        Raises:
            ExperimentError: At least a full grid of sample results must be registered to predict
                             the best configuration.

        Returns:
            tuple: Two elements tuple with a Dictionary of best parameters values and related uncertainty of
                   minimum cost.
        """
        if len(self.optimizer.yi) < self.optimizer.n_initial_points_:
            raise ExperimentError(
                "Not enough samples are registered (at least {} are required".format(
                    self.optimizer.n_initial_points_
                )
            )
        best_config = {}
        res = self.optimizer.get_result()
        best_point = skopt.expected_minimum(res, 100)[0]
        for i, p in enumerate(self.params_space):
            best_config[p.name] = best_point[i]
        fmin, std = res.models[0].predict(np.array([best_point]), return_std=True)
        costs = [s.cost for s in self.get_samples_list()]
        uncertainty = 4 * std / (max(costs) - min(costs))
        log.info("Best config for {} Experiment retrived.".format(self.name))
        return best_config, uncertainty[0], fmin[0]

    def to_dataframe(self) -> pd.DataFrame:
        """Generate a pandas dataframe from designs and results.

        Returns:
            pd.DataFrame: dataframe descripting all designs.
        """
        designs = self.get_designs()
        if len(designs) == 0:
            raise ExperimentError("Designs must be generated first to be exported.")
        results = self.get_samples_results_df()
        grid_ids_list = np.array([])
        for i, g in enumerate(self.sample_grid_list):
            grid_ids_list = np.append(grid_ids_list, i * np.ones(len(g.samples_list)))
        grid_ids = pd.DataFrame(data=grid_ids_list, columns=["sample_grid_id"])
        df = pd.concat([designs, results, grid_ids], axis=1)
        return df

    def from_dataframe(
        self,
        samples_df: pd.DataFrame,
        overwrite_samples: bool = True,
        clip_to_space: bool = True,
        infer_space: bool = True,
    ):
        """Import experiment dataframe into current Experiment object.
           Existing samples can be overwritten.

        Args:
            file_path (str): File path of csv file
            overwrite_samples (True): overwrite existing samples with csv samples data.
            clip_to_space (True): clip parameters values to space while importing.
            infer_space (True): Try to infer the parametric space from designs (experimental).
        """
        designs_df = samples_df.drop(
            columns=["print_time", "quality", "cost", "sample_grid_id"]
        )
        grid_ids = samples_df["sample_grid_id"].unique().astype(int).tolist()
        if overwrite_samples:
            self.sample_grid_list = []
        if infer_space:
            self.sample_grid_list = []
            self.params_space = self.__infer_param_space_from_designs_df(designs_df)
        if clip_to_space:
            designs_df = self.__clip_designs_df_to_parameters_space(designs_df)
        for i in grid_ids:
            grid_designs_df = designs_df[samples_df["sample_grid_id"] == i].reset_index(
                drop=True
            )
            quality = samples_df[samples_df["sample_grid_id"] == i][
                "quality"
            ].reset_index(drop=True)
            self.create_new_sample_grid(len(grid_designs_df))
            self.sample_grid_list[-1].designs = grid_designs_df
            self.sample_grid_list[-1].quality_list = quality
        pass

    def export_csv(self, file_path: str) -> None:
        """Export designs and current state of results to a csv file.

        Args:
            file_path (str): file_path

        Returns:
            None:
        """
        self.to_dataframe().to_csv(file_path)
        pass

    def import_csv(
        self,
        file_path: str,
        overwrite_samples: bool = True,
        clip_to_space: bool = True,
        infer_space: bool = True,
    ) -> None:
        """Import csv experiment data to current Experiment object.
           Existing samples can be overwritten.

        Args:
            file_path (str): File path of csv file
            overwrite_samples (True): overwrite existing samples with csv samples data.
            clip_to_space (True): clip parameters values to space while importing.
            infer_space (True): Try to infer the parametric space from designs (experimental).
        """
        samples_df = pd.read_csv(file_path, index_col=0)
        self.from_dataframe(
            samples_df=samples_df,
            overwrite_samples=overwrite_samples,
            clip_to_space=clip_to_space,
            infer_space=infer_space,
        )
        pass

    def __infer_param_space_from_designs_df(self, designs: pd.DataFrame) -> skopt.Space:
        """Infer parameters space from a given designs DataFrame

        Args:
            designs (DataFrame): design DataFrame with parameters as column names.

        Returns:
            skopt.Space: space of parameters
        """
        params = []
        if self.is_first_layer:
            # drop useless columns
            designs.drop(columns=[])
        for col in designs.columns:
            # infer data type and set param
            s = designs[col]
            col_type = str(s.dtype)
            if "int" in col_type:
                p = skopt.space.Integer(name=col, low=s.min(), high=s.max())
            elif "float" in col_type:
                p = skopt.space.Real(name=col, low=s.min(), high=s.max())
            params.append(p)
        return skopt.Space(params)

    def __clip_designs_df_to_parameters_space(
        self, designs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """clip designs dataframe values to parameters space bounds.

        Args:
            designs_df (pd.DataFrame): designs dataframe

        Returns:
            pd.DataFrame: clipped designs dataframe
        """
        for p in self.params_space:
            low_bound = p.low
            high_bound = p.high
            designs_df[p.name] = designs_df[p.name].clip(
                lower=low_bound, upper=high_bound
            )
        return designs_df

    def write_validation_sample(self, params: dict, output_file: str) -> None:
        """Write gcode for validation sample.

        Args:
            params (dict): Sample parameters for validation sample.
            output_file (str): Path for validation sample gcode.
        """
        sample = Sample(
            is_first_layer=self.is_first_layer,
            printer=self.printer,
            filament=self.filament,
            input_file=self.sample_file,
        )
        sample.erase_start_end_gcode()
        init_gcode = self.sample_grid_list[-1].init_gcode
        end_gcode = self.sample_grid_list[-1].end_gcode
        for key, val in params.items():
            sample.set_param(name=key, value=val)
        gcode = init_gcode + sample.gcode + end_gcode
        with open(output_file, "w") as file:
            file.write(gcode)
        pass
