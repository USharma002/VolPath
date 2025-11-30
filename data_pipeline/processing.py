import numpy as np


class DataProcessor:
    """Handles data transformations and scalar generation."""

    DENSITY_FIELDS = {'Density', 'Rho', 'rho'}
    UTHERM_FIELDS = {'InternalEnergy', 'Utherm'}
    ELECTRON_FIELD = 'ElectronAbundance'
    _DENSITY_FIELDS_LOWER = {name.lower() for name in DENSITY_FIELDS}
    _UTHERM_FIELDS_LOWER = {name.lower() for name in UTHERM_FIELDS}

    @staticmethod
    def apply_transform(field_name, field_data, transform, polydata=None, header=None):
        transform = transform or 'Linear'

        if transform == 'Linear':
            return field_data
        if transform == 'Log':
            return DataProcessor._log_shifted(field_data)
        if transform == 'Magnitude':
            return np.linalg.norm(field_data, axis=1)
        if transform == 'Log Magnitude':
            mag = np.linalg.norm(field_data, axis=1)
            return np.log10(mag + 1e-10)
        if transform == 'X Component':
            return field_data[:, 0]
        if transform == 'Y Component':
            return field_data[:, 1]
        if transform == 'Z Component':
            return field_data[:, 2]
        if transform == 'Normalized':
            arr = np.asarray(field_data, dtype=np.float64)
            min_val = np.min(arr)
            max_val = np.max(arr)
            if max_val <= min_val:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
        if transform == 'Log Density (Arepo)':
            return DataProcessor._arepo_log(field_data)
        if transform == 'Log InternalEnergy (Arepo)':
            return DataProcessor._arepo_log(field_data)
        if transform == 'Kelvin (InternalEnergy)':
            kelvin = DataProcessor._convert_internal_energy_to_kelvin(
                field_data, polydata, header
            )
            return kelvin
        if transform == 'Kelvin Log10':
            kelvin = DataProcessor._convert_internal_energy_to_kelvin(
                field_data, polydata, header
            )
            return DataProcessor._arepo_log(kelvin)
        return field_data

    @staticmethod
    def get_available_transforms(field_name, field_data, polydata=None, header=None):
        transforms = []
        if DataProcessor._is_vector(field_data):
            transforms.extend(['Magnitude', 'Log Magnitude', 'X Component', 'Y Component', 'Z Component'])
        else:
            transforms.extend(['Linear', 'Log', 'Normalized'])
            if DataProcessor._is_density_field(field_name):
                transforms.insert(1, 'Log Density (Arepo)')
            if DataProcessor._is_internal_energy_field(field_name):
                transforms.insert(1, 'Log InternalEnergy (Arepo)')
                if DataProcessor._has_kelvin_support(header):
                    transforms.append('Kelvin (InternalEnergy)')
                    transforms.append('Kelvin Log10')
        return transforms

    @staticmethod
    def _is_vector(field_data):
        return field_data.ndim > 1 and field_data.shape[1] >= 3

    @staticmethod
    def _is_density_field(field_name):
        return (field_name or '').lower() in DataProcessor._DENSITY_FIELDS_LOWER

    @staticmethod
    def _is_internal_energy_field(field_name):
        return (field_name or '').lower() in DataProcessor._UTHERM_FIELDS_LOWER

    @staticmethod
    def _has_kelvin_support(header):
        if not header:
            return False
        return 'UnitEnergy_in_cgs' in header and 'UnitMass_in_g' in header

    @staticmethod
    def _log_shifted(values):
        arr = np.asarray(values, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        min_val = finite.min() if finite.size else 0.0
        shift = 1e-10
        if min_val <= 0:
            shift += -min_val + 1e-10
        return np.log10(arr + shift)

    @staticmethod
    def _arepo_log(values):
        arr = np.asarray(values, dtype=np.float64)
        arr = np.clip(arr, 1e-30, None)
        return np.log10(arr) + 10.0

    @staticmethod
    def _flatten_scalar(field):
        arr = np.asarray(field, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        reshaped = arr.reshape(arr.shape[0], -1)
        return reshaped[:, 0]

    @staticmethod
    def _convert_internal_energy_to_kelvin(field_data, polydata, header):
        if not DataProcessor._has_kelvin_support(header):
            raise ValueError("Kelvin conversion requires unit metadata from the snapshot header")

        utherm = DataProcessor._flatten_scalar(field_data)
        if polydata and DataProcessor.ELECTRON_FIELD in polydata:
            nelec = DataProcessor._flatten_scalar(polydata[DataProcessor.ELECTRON_FIELD])
        else:
            nelec = np.zeros_like(utherm)

        if nelec.shape[0] != utherm.shape[0]:
            raise ValueError("ElectronAbundance field size does not match InternalEnergy array")

        gamma = 5.0 / 3.0
        hmassfrac = 0.76
        mass_proton = 1.672622e-24
        boltzmann = 1.380650e-16
        unit_energy = float(header.get('UnitEnergy_in_cgs', 1.0))
        unit_mass = float(header.get('UnitMass_in_g', 1.0)) or 1.0

        factor1 = 4.0 * mass_proton
        factor2 = 1.0 + 3.0 * hmassfrac
        factor3 = (gamma - 1.0) / boltzmann * unit_energy / unit_mass

        meanmolwt = factor1 / (factor2 + 4.0 * hmassfrac * nelec)
        return factor3 * utherm * meanmolwt


__all__ = ["DataProcessor"]
