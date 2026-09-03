"""pytest tests for module definitions/config_file.py, focused on the phase 3 (gdcluster) section."""
import configparser

from gausspyplus.definitions import config_file as cf


def test_make_writes_a_separate_phase_3_section(tmp_path):
    cf.make(all_keywords=True, output_directory=tmp_path, filename="gausspy+.ini")

    config = configparser.ConfigParser()
    config.read(tmp_path / "gausspy+.ini")

    assert "phase 3" in config
    assert "gdcluster_min_comp" in config["phase 3"]
    assert "gdcluster_neighbor_radius" in config["phase 3"]
    assert "refit_gdcluster" in config["phase 3"]

    #  the gdcluster settings must not also be duplicated under 'spatial fitting'
    assert "gdcluster_min_comp" not in config["spatial fitting"]
    assert "refit_gdcluster" not in config["spatial fitting"]


def test_spatial_fitting_reads_phase_3_section(tmp_path):
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    ini_path = tmp_path / "gausspy+.ini"
    ini_path.write_text(
        "[DEFAULT]\n"
        "\n"
        "[spatial fitting]\n"
        "\n"
        "[phase 3]\n"
        "refit_gdcluster = True\n"
        "gdcluster_min_comp = 9\n"
        "gdcluster_neighbor_radius = 8.32\n"
    )

    sp = SpatialFitting(config_file=str(ini_path))

    assert sp.refit_gdcluster is True
    assert sp.gdcluster_min_comp == 9
    assert sp.gdcluster_neighbor_radius == 8.32


def test_spatial_fitting_backward_compatible_with_missing_phase_3_section(tmp_path):
    """Config files written before phase 3 existed have no '[phase 3]' section at all; the class
    must fall back to the dataclass defaults instead of raising."""
    from gausspyplus.spatial_fitting.spatial_fitting import SpatialFitting

    ini_path = tmp_path / "gausspy+.ini"
    ini_path.write_text("[DEFAULT]\n\n[spatial fitting]\nmean_separation = 3.0\n")

    sp = SpatialFitting(config_file=str(ini_path))

    assert sp.mean_separation == 3.0  # picked up from '[spatial fitting]' as before
    assert sp.refit_gdcluster is False  # falls back to the dataclass default
    assert sp.gdcluster_min_comp == 6  # falls back to the dataclass default
