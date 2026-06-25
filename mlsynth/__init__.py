from importlib.metadata import version as _version, PackageNotFoundError
try:
    __version__ = _version("mlsynth")
except PackageNotFoundError:  # not installed (e.g. run from a source checkout)
    __version__ = "0.1.2"

from .estimators.tssc import TSSC ## Check
from .estimators.fma import FMA ## Check
from .estimators.pda import PDA ## Check
from .estimators.fdid import FDID ## Check
from .estimators.clustersc import CLUSTERSC ## Check
from .estimators.proximal import PROXIMAL ## Check
from .estimators.fscm import FSCM ## Check
from .estimators.scmo import SCMO
from .estimators.scta import SCTA
from .estimators.si import SI ## Check
from .estimators.nsc import NSC # Check
from .estimators.sdid import SDID # Check
from .estimators.musc import MUSC                              # noqa: F401
from .estimators.masc import MASC                              # noqa: F401
from .estimators.shc import SHC # Check
from .estimators.laxscm import RESCM # Check
from .estimators.scexp import MAREX
from .estimators.msqrt import MSQRT
from .estimators.ssc import SSC
from .estimators.rmsi import RMSI
from .estimators.spotsynth import SPOTSYNTH
from .estimators.lexscm import LEXSCM
from .estimators.syndes import SYNDES
from .utils.syndes_helpers.power import SYNDESPower, power_analysis
from .utils.syndes_helpers.select import SYNDESRecommendation, recommend_syndes
from .utils.design_compare import (
    DesignComparison,
    DesignSpec,
    compare_methods,
    compare_pareto,
    from_geolift,
    from_syndes,
    plot_compare_pareto,
)
from .estimators.spcd import SPCD
from .estimators.tasc import TASC
from .estimators.cmbsts import CMBSTS
from .estimators.sbc import SBC
from .estimators.bvss import BVSS
from .estimators.mlsc import MLSC
from .estimators.seq_sdid import SequentialSDID
from .estimators.ppscm import PPSCM
from .estimators.sparse_sc import SparseSC
from .estimators.microsynth import MicroSynth
from .estimators.siv import SIV
from .estimators.orthsc import ORTHSC
from .estimators.dsc import DSC
from .estimators.dscar import DSCAR
from .estimators.spsydid import SpSyDiD
from .estimators.iscm import ISCM
from .estimators.vanillasc import VanillaSC
from .estimators.spillsynth import SPILLSYNTH
from .estimators.ctsc import CTSC
from .estimators.snn import SNN
from .estimators.mcnnm import MCNNM
from .estimators.pangeo import PANGEO
from .estimators.hsc import HSC
from .estimators.geolift import GEOLIFT
from .estimators.multicellgeolift import MULTICELLGEOLIFT
from .estimators.rolldid import ROLLDID
from .utils.spcd_helpers.plotter import (
    plot_spcd_design,
    plot_mde_bars,
    plot_power_curves,
    plot_detectability,
)
from .utils.truncated_history import (
    truncated_history,
    TruncatedHistoryResult,
    TruncatedHistoryWindow,
)
from .spec import save_spec, load_spec

__all__ = [
    "save_spec",
    "load_spec",
    "plot_spcd_design",
    "plot_mde_bars",
    "plot_power_curves",
    "plot_detectability",
    "GEOLIFT",
    "MULTICELLGEOLIFT",
    "ROLLDID",
    "HSC",
    "TSSC",
    "FMA",
    "PDA",
    "FDID",
    "CLUSTERSC",
    "PROXIMAL",
    "FSCM",
    "SCMO",
    "SCTA",
    "SI",
    "NSC",
    "MUSC",
    "MASC",
    "SDID",
    "SHC",
    "RESCM",
    "MAREX",
    "LEXSCM",
    "SPCD",
    "TASC",
    "CMBSTS",
    "SBC", "BVSS",
    "MLSC",
    "MSQRT",
    "SSC",
    "RMSI",
    "SPOTSYNTH",
    "SequentialSDID",
    "PPSCM",
    "SparseSC",
    "MicroSynth",
    "SIV",
    "ORTHSC",
    "SYNDES",
    "SYNDESPower",
    "power_analysis",
    "SYNDESRecommendation",
    "recommend_syndes",
    "DesignSpec",
    "DesignComparison",
    "compare_methods",
    "compare_pareto",
    "from_syndes",
    "from_geolift",
    "plot_compare_pareto",
    "DSC",
    "SpSyDiD",
    "ISCM",
    "VanillaSC",
    "truncated_history", "TruncatedHistoryResult", "TruncatedHistoryWindow",
    "DSCAR",
    "SPILLSYNTH",
    "CTSC",
    "SNN",
    "MCNNM",
    "PANGEO",
]
