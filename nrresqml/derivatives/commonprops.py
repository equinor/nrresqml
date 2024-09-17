import dataclasses


@dataclasses.dataclass
class CommonProp:
    name: str
    lower: float
    upper: float


def lookup_common_prop(name: str) -> CommonProp:
    return [s for s in STANDARD_PROPS if s.name == name][0]


STANDARD_PROPS = [
    # First iteration
    CommonProp('DXX01', 0.0, 50.0),
    CommonProp('DXX02', 0.0, 50.0),
    CommonProp('DXX03', 0.0, 50.0),
    CommonProp('DXX04', 0.0, 50.0),
    CommonProp('DXX05', 0.0, 50.0),
    CommonProp('d50_per_sedclass', 0.0, 50.0),
    CommonProp('diameter', 0.0, 50.0),
    CommonProp('fraction', 0.0, 1.0),
    CommonProp('sorting', 0.0, 1.0),
    CommonProp('Sed1_mass', 0.0, 1e6),
    CommonProp('Sed2_mass', 0.0, 1e6),
    CommonProp('Sed3_mass', 0.0, 1e6),
    CommonProp('Sed4_mass', 0.0, 1e6),
    CommonProp('Sed5_mass', 0.0, 1e6),
    CommonProp('Sed6_mass', 0.0, 1e6),
    CommonProp('Sed1_volfrac', 0.0, 1.0),
    CommonProp('Sed2_volfrac', 0.0, 1.0),
    CommonProp('Sed3_volfrac', 0.0, 1.0),
    CommonProp('Sed4_volfrac', 0.0, 1.0),
    CommonProp('Sed5_volfrac', 0.0, 1.0),
    CommonProp('Sed6_volfrac', 0.0, 1.0),
    CommonProp('Porosity', 0.0, 1.0),
    CommonProp('porosity', 0.0, 1.0),
    CommonProp('permeability', 0.0, 1.0),
]
