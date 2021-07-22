# -*- coding: utf-8 -*-
"""sliceoptim module to integrate Printer, Filament and Experiment objects in a database.
"""

__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "apache 2.0"

from sliceoptim import core
from tinydb import TinyDB, where
from pathlib import Path
import pickle
import os


class Database:
    def __init__(self, folder_path: str) -> None:
        """Class for database management of Printer, Filaments and Experiments.

        Args:
            folder_path (str): path for the database.
        """
        self.folder_path = Path(folder_path)
        self.tdb = TinyDB(self.folder_path / "sliceoptim.db")
        self.filament_table = self.tdb.table("filament")
        self.printer_table = self.tdb.table("printer")
        self.experiment_table = self.tdb.table("experiment")
        pass

    def __raise_error_if_already_exists(self, table, name: str):
        """Raises an error if a database element already exists.

        Args:
            table ([type]): TinyDB table.
            name (str): element name.

        Raises:
            ValueError: Error if element already exists.
        """
        element = table.search(where("name") == name)
        if len(element) > 0:
            raise ValueError(
                "Element from table {} with name {} already exists!".format(
                    str(table), name
                )
            )
        pass

    def add_filament(self, filament: core.Filament):
        """Adds a filament object to database.

        Args:
            filament (core.Filament): Filament object to append.
        """
        self.__raise_error_if_already_exists(self.filament_table, filament.name)
        self.filament_table.insert(filament.__dict__)
        pass

    def get_filament(self, name: str) -> core.Filament:
        """Gets a Filament object from database with its name.

        Args:
            name (str): Filament name.

        Raises:
            ValueError: Error if Filament name does not exist.

        Returns:
            core.Filament: Stored Filament object.
        """
        try:
            filament = self.filament_table.search(where("name") == name)[0]
        except IndexError:
            raise ValueError("Filament {} does no exist.".format(name)) from None
        return core.Filament(**filament)

    def get_filament_names(self) -> list:
        """Returns a list of stored Filament names.

        Returns:
            list: List of Filament names.
        """
        return [f["name"] for f in self.filament_table.all()]

    def delete_filament(self, name: str):
        """Delete a Filament in database.

        Args:
              name (str): Name of Filament to delete.
        """
        self.filament_table.remove(where("name") == name)
        pass

    def add_printer(self, printer: core.Printer):
        """Adds a Printer object to the database.

        Args:
            printer (core.Printer): Printer object to add.
        """
        self.__raise_error_if_already_exists(self.printer_table, printer.name)
        self.printer_table.insert(printer.__dict__)
        pass

    def get_printer(self, name: str) -> core.Printer:
        """Get a Printer object for database.

        Args:
            name (str): Name of Printer to retreive.

        Raises:
            ValueError: Error if Printer name not in database.

        Returns:
            core.Printer: Printer object.
        """
        try:
            printer = self.printer_table.search(where("name") == name)[0]
        except IndexError:
            raise ValueError("printer {} does no exist.".format(name)) from None
        return core.Printer(**printer)

    def get_printer_names(self) -> list:
        """Returns a list of Printer names stored in database.

        Returns:
            list: List of Printer names.
        """
        return [p["name"] for p in self.printer_table.all()]

    def delete_printer(self, name: str):
        """Delete a Printer in database.

        Args:
            name (str): Name of Printer to delete.
        """
        self.printer_table.remove(where("name") == name)
        pass

    def add_experiment(self, experiment: core.Experiment):
        """Adds an Experiment object to database.

        Args:
            experiment (core.Experiment): Experiment object.
        """
        self.__raise_error_if_already_exists(self.experiment_table, experiment.name)
        dump_filepath = (self.folder_path / experiment.name).with_suffix(".exp")
        with open(dump_filepath, "wb") as exp_file:
            pickle.dump(experiment, exp_file)
        self.experiment_table.insert(
            {"name": experiment.name, "filepath": str(dump_filepath)}
        )
        pass

    def get_experiment_filepath(self, name: str) -> str:
        """Get the path of a given saved Experiment.

        Args:
            name (str): Experiment name.

        Raises:
            ValueError: Error if Experiment not existing.

        Returns:
            str: Path of Experiment serialized object.
        """
        try:
            experiment_filepath = self.experiment_table.search(where("name") == name)[
                0
            ]["filepath"]
        except IndexError:
            raise ValueError("experiment {} does no exist.".format(name)) from None
        return experiment_filepath

    def get_experiment(self, name: str) -> core.Experiment:
        """Get a saved Experiment object.

        Args:
            name (str): Experiment name.

        Returns:
            core.Experiment: saved Experiment object.
        """
        filepath = self.get_experiment_filepath(name)
        with open(filepath, "rb") as exp_file:
            experiment = pickle.load(exp_file)
        return experiment

    def get_experiment_names(self) -> list:
        """Returns a list of saved Experiment names.

        Returns:
            list: List of saved Experiments names.
        """
        return [e["name"] for e in self.experiment_table.all()]

    def delete_experiment(self, name: str):
        """Delete a saved Experiment.

        Args:
            name (str): Experiment name.
        """
        filepath = self.get_experiment_filepath(name)
        os.remove(filepath)
        self.experiment_table.remove(where("name") == name)
        pass

    def update_experiment(self, experiment: core.Experiment):
        """Overwrites a saved experiment.

        Args:
            experiment (core.Experiment): Experiment object.
        """
        self.delete_experiment(name=experiment.name)
        self.add_experiment(experiment=experiment)
        pass
