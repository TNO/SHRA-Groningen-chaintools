import os
import xarray as xr
import tempfile
import yaml
from unittest import TestCase
import chaintools.tools_configuration as cfg
import chaintools.tools_xarray as xf

class IntegrationTests(TestCase):

    test_config_file = './tests/res/test_config.yml'
    test_config = {
        'data_sources':
            {'dummy_data_in': {'type': 'xarray_dataset',
                               'path': ['./tests/res/', 'GMM_config_testing.h5'],
                               'group': 'GMM-V7'}},
        'data_sinks':
            {'dummy_data_out': {'type': 'xarray_dataset',
                                'path': ['./tests/res/', 'GMM_tables_testing.h5'],
                                'group': 'GMM-V7'}},
        'dimensions':
            {'dimension_1': {'length': 10, 'interval': [1.4, 7.1]},
             'dimension_2': {'length': 10, 'sequence_spacing': 'log', 'interval': [3.0, 40.0], 'units': 'km'},
             'dimension_3': {'length': 10}, 'dimension_4': {'length': 10}},
        'coordinates':
            {'dimension_3_alt': {'dim': 'dimension_3', 'sequence_spacing': 'log',
                                 'interval': [1e-05, 10.0], 'units': 'donkeys'},
             'dimension_3_alt_bis': {'dim': 'dimension_3', 'sequence_spacing': 'log',
                                     'interval': [1e-05, 10.0], 'units': 'donkeys_squared', 'multiplier': 981.0},
             'dimension_4_alt': {'dim': 'dimension_4', 'sequence_spacing': 'log',
                                 'interval': [0.001, 2.0], 'units': 'donkeys'}},
        'n_workers': 1
    }

    def test_configuration(self):

        # arrange
        # -------
        module = 'dummy_module'
        expected_config = self.test_config

        # act
        # ---
        config = cfg.configure(self.test_config_file, module_name=module)

        # assert
        # ------
        IntegrationTests.assertDictEqual(self, expected_config, config)

    def test_prepare_ds(self):

        # arrange
        # -------
        dataset_empty_expected = xr.load_dataset('./tests/res/test_empty_dataset.h5', engine='h5netcdf')
        #
        # act
        # ---
        dataset_empty = xf.prepare_ds(self.test_config)

        # assert
        # ------
        xr.testing.assert_identical(dataset_empty_expected, dataset_empty)


    @staticmethod
    def build_config_file_for_module(self, config: dict, dir: str, module: str, config_name: str= 'temp_config.yml') -> str:
        """
        Write and save a new configuration .yaml file for a specific module.
        :param config: Dictionary with configuration for a specific module
        :param dir: Directory where the configuration files will be stored
        :param module: Name of the module
        :param config_name: Optional, name of the configuration file.
        :return: Returns the path to the new configuration file.
        """

        temp_module_config = {'modules': {module: config}}
        temp_yml_path = os.path.join(dir, config_name)
        with open(temp_yml_path, 'w') as f:
            yaml.dump(temp_module_config, f)

        return temp_yml_path
