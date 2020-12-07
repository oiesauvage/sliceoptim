# -*- coding: utf-8 -*-

__author__ = "Nils Artiges"
__copyright__ = "Nils Artiges"
__license__ = "mit"

from sliceoptim import core, samples
from tinydb import TinyDB, where
from pathlib import Path
import pickle
import os


class Database:
    def __init__(self, folder_path: str) -> None:
        self.folder_path = Path(folder_path)
        self.tdb = TinyDB(self.folder_path / "sliceoptim.db")
        self.filament_table = self.tdb.table("filament")
        self.printer_table = self.tdb.table("printer")
        self.experiment_table = self.tdb.table("experiment")
        pass

    def __raise_error_if_already_exists(self, table, name: str):
        element = table.search(where("name") == name)
        if len(element) > 0:
            raise ValueError(
                "Element from table {} with name {} already exists!".format(
                    str(table), name
                )
            )
        pass

    def add_filament(self, filament: core.Filament):
        self.__raise_error_if_already_exists(self.filament_table, filament.name)
        self.filament_table.insert(filament.__dict__)
        pass

    def get_filament(self, name: str) -> core.Filament:
        try:
            filament = self.filament_table.search(where("name") == name)[0]
        except IndexError:
            raise ValueError("Filament {} does no exist.".format(name)) from None
        return core.Filament(**filament)

    def get_filament_names(self) -> list:
        return [f["name"] for f in self.filament_table.all()]

    def delete_filament(self, name: str):
        self.filament_table.remove(where("name") == name)
        pass

    def add_printer(self, printer: core.Printer):
        self.__raise_error_if_already_exists(self.printer_table, printer.name)
        self.printer_table.insert(printer.__dict__)
        pass

    def get_printer(self, name: str) -> core.Printer:
        try:
            printer = self.printer_table.search(where("name") == name)[0]
        except IndexError:
            raise ValueError("printer {} does no exist.".format(name)) from None
        return core.Printer(**printer)

    def get_printer_names(self) -> list:
        return [p["name"] for p in self.printer_table.all()]

    def delete_printer(self, name: str):
        self.printer_table.remove(where("name") == name)
        pass

    def add_experiment(self, experiment: core.Experiment):
        self.__raise_error_if_already_exists(self.experiment_table, experiment.name)
        dump_filepath = (self.folder_path / experiment.name).with_suffix(".exp")
        with open(dump_filepath, "wb") as exp_file:
            pickle.dump(experiment, exp_file)
        self.experiment_table.insert(
            {"name": experiment.name, "filepath": str(dump_filepath)}
        )
        pass

    def get_experiment_filepath(self, name: str) -> str:
        try:
            experiment_filepath = self.experiment_table.search(where("name") == name)[
                0
            ]["filepath"]
        except IndexError:
            raise ValueError("experiment {} does no exist.".format(name)) from None
        return experiment_filepath

    def get_experiment(self, name: str) -> core.Experiment:
        filepath = self.get_experiment_filepath(name)
        with open(filepath, "rb") as exp_file:
            experiment = pickle.load(exp_file)
        return experiment

    def get_experiment_names(self) -> list:
        return [e["name"] for e in self.experiment_table.all()]

    def delete_experiment(self, name: str):
        filepath = self.get_experiment_filepath(name)
        os.remove(filepath)
        self.experiment_table.remove(where("name") == name)
        pass

    def update_experiment(self, experiment: core.Experiment):
        self.delete_experiment(name=experiment.name)
        self.add_experiment(experiment=experiment)
        pass
