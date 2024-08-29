import pathlib
import pydap.client
import webob.exc
from pydap.model import DatasetType
from typing import List, Union, Optional

import h5py
import numpy as np

from nrresqml.derivatives import commonprops
from nrresqml.derivatives.ijkgridcreator import IjkGridCreator, IjkGridCreationError
from nrresqml.factories.energetics import create_hdf5_reference
from nrresqml.factories.resqml.properties import create_continuous_property, create_categorical_property
from nrresqml.structures.energetics import AbstractObject
from nrresqml.structures.resqml.properties import (
    ContinuousProperty,
    AbstractValuesProperty,
)


class AdaptorError(Exception):
    pass


class Hdf5ResQmlAdaptor:
    """
    Interface class for converting arbitrary data into an hdf5 file suited for ResQml, and its corresponding ResQml data
    objects
    """

    def create_objects(self) -> List[AbstractObject]:
        raise NotImplementedError('Method not implemented by subclass')

    def dump_h5_file(self, filename: pathlib.Path):
        raise NotImplementedError('Method not implemented by subclass')

    def h5_base_name(self) -> str:
        raise NotImplementedError('Method not implemented by subclass')


def _open_delft3d_path(p: str) -> Union[h5py.File, DatasetType]:
    if p.startswith('http'):
        return pydap.client.open_url(p)
    else:
        return h5py.File(p, mode='r')


class Delft3DResQmlAdaptor(Hdf5ResQmlAdaptor):
    _archel_map = {
        0: 'Inactive',
        1: 'Active Channel',
        2: 'Channel Fill',
        3: 'Delta Top Background',
        4: 'Mouth Bar',
        5: 'Delta Front Background',
        6: 'Pro-delta',
    }
    _subenviron_map = {
        0: 'Background',
        1: 'Delta Top',
        2: 'Delta Front',
        3: 'Pro-delta',
    }
    # HDF5 keys
    _delft3d_archel_key = 'archel'
    _delft3d_subenv_key = 'subenv'
    _resqml_archel_key = 'archel'
    _resqml_subenv_key = 'subenv'
    # Continuous attributes and their natural bounds. We can compute the bounds per data set, but it is not obvious from
    # the ResQML specification that it is required. Instead, we choose to use some natural bounds based on what the
    # properties represents. This can save a significant amount of computation time.

    def __init__(self, d3_file: str, archel_file: str) -> None:
        self._d3_file = _open_delft3d_path(d3_file)
        try:
            self._archel_file = _open_delft3d_path(archel_file)
        except (webob.exc.HTTPError, OSError):
            print(f'Architectural elements/sub-environment not defined: Failed to open file {archel_file}')
            self._archel_file = None

        # Grid handling
        try:
            self._grid_creator = IjkGridCreator(self._d3_file)
        except IjkGridCreationError as e:
            raise AdaptorError(f'Failed to create grid from {d3_file} with the following error:\n  ' + str(e))

        # Continuous properties
        self._continuous_properties = []
        for pn in commonprops.STANDARD_PROPS:
            try:
                self._continuous_properties.append(self._d3_file[pn.name])
                print(f'Included continuous property {pn.name}')
            except KeyError:
                print(f'Did not find property {pn.name}')
        print('')

        # Check if there is a vfractions property. If yes, then this is probably a newer format
        self._volume_fractions = Delft3DResQmlAdaptor._try_find_vfractions_array(self._d3_file)

    def create_objects(self) -> List[AbstractObject]:
        ref = create_hdf5_reference()
        ijk = self._grid_creator.ijk_representation(ref)
        # Create continuous properties
        props: List[AbstractValuesProperty] = [
            Delft3DResQmlAdaptor._create_continuous_property(p, ijk, ref)
            for p in self._continuous_properties
        ]
        if self._volume_fractions is not None:
            props += [
                Delft3DResQmlAdaptor._create_continuous_property(f'Sed{i}_volfrac', ijk, ref)
                for i in range(1, self._volume_fractions.shape[1] + 1)
            ]

        # Create categorical properties
        # Architectural elements
        props.append(create_categorical_property(
            'Architectural element', self._resqml_archel_key, ijk, ref, self._archel_map
        ))
        # Sub-environment
        props.append(create_categorical_property(
            'Subenvironment', self._resqml_subenv_key, ijk, ref, self._subenviron_map
        ))

        return [ijk, ijk.Geometry.LocalCrs, ref] + props

    def dump_h5_file(self, filename: pathlib.Path):
        out = h5py.File(filename, 'w')
        for p in self._continuous_properties:
            out.copy(p, p.name)

        if self._volume_fractions is not None:
            print('Splitting vfractions array')
            chunks = (
                self._volume_fractions.chunks[0],
                self._volume_fractions.chunks[2],
                self._volume_fractions.chunks[3],
            )
            for i in range(self._volume_fractions.shape[1]):
                out.create_dataset(
                    f'Sed{i+1}_volfrac',
                    data=self._volume_fractions[:, i, :, :],
                    compression=self._volume_fractions.compression,
                    compression_opts=self._volume_fractions.compression_opts,
                    chunks=chunks,
                )

        # Define temporary function to extract archel data
        def _copy_archel_data(source, target):
            if self._archel_file is None or source not in self._archel_file:
                nx, ny, nz = self._grid_creator.pillars.shape[-3:]
                out.create_dataset(
                    target, data=np.zeros((nz, nx, ny), dtype=np.float32), compression='gzip'
                )
            else:
                out.copy(self._archel_file[source], target)

        # Architectural elements data set (zero-array if key does not exist)
        _copy_archel_data(self._delft3d_archel_key, self._resqml_archel_key)

        # Sub-environment data set (zero-array if key does not exist)
        _copy_archel_data(self._delft3d_subenv_key, self._resqml_subenv_key)

        # Dump data
        self._grid_creator.dump_grid_data(out)

    def h5_base_name(self) -> str:
        return 'Delft3d.h5'

    @staticmethod
    def _try_find_vfractions_array(
        d3_file: Union[h5py.File, DatasetType]
    ) -> Optional[h5py.Dataset]:
        volfrac = d3_file.get('vfractions')
        if volfrac is None:
            return
        if 'Sed1_volfrac' in d3_file:
            print(
                f'Both SedX_volfrac and vfractions are present. vfractions will be ignored'
            )
            return
        if volfrac.ndim != 4:
            print(
                f'Wrong volume fractions array dimension: {volfrac.ndim} (expected'
                f' 4). vfractions will be ignored'
            )
            return
        if volfrac.shape[1] != 6:
            print(
                f'Wrong volume fractions array shape: {volfrac.shape}. Second'
                f' dimension should have length 6. vfractions will be ignored'
            )
            return
        return volfrac

    @staticmethod
    def _create_continuous_property(prop, ijk, ref) -> ContinuousProperty:
        if isinstance(prop, str):
            pn = prop
            long_name = prop
        else:
            # Workaround to support both HDF5 and pydap when finding 'long_name':
            try:
                long_name = prop.attrs['long_name']
                # sometimes models returns bytes, so we need to decode it
                if isinstance(long_name, bytes):
                    long_name = long_name.decode('utf-8')
            except AttributeError:
                long_name = prop.attributes['long_name']
            pn = prop.name.strip('/')
        return create_continuous_property(
            long_name,
            pn,
            commonprops.lookup_common_prop(pn).lower,
            commonprops.lookup_common_prop(pn).upper,
            ijk,
            ref,
        )

