import os
import typing as t
import pandas as pd

from pydantic.dataclasses import dataclass
from pydantic import Field

from pyfm import utils
from pyfm.tasks.hadrons.types import HadronsInput
import pyfm.tasks.hadrons.modules as hadmods
from pyfm.domain import (
    SimpleConfig,
    Outfile,
    MassDict,
)
from pyfm.tasks.register import register_task
from pyfm.tasks.hadrons.types import LanczosParams


@dataclass(frozen=True)
class EpackConfig(SimpleConfig):
    mass: MassDict
    eigs: int
    eig: Outfile
    eigdir: Outfile
    eval: Outfile
    load: bool = True
    lanczos: LanczosParams | None = None
    action_name: str | None = None
    low_modes_name: str | None = None
    multifile: bool = False
    save_eigs: bool = False
    save_evals: bool = True
    # Checkpoint the IRL eigenpack in QUDA/MILC partfile format so a restarted
    # job can read it back instead of re-running the Lanczos.
    checkpoint: bool = False
    checkpoint_single_prec: bool = True
    eig_quda: Outfile | None = None

    @property
    def checkpoint_stem(self) -> str:
        """Partfile stem; each rank writes <stem>.vol<NNNN>."""
        return self.eig_quda.filename

    @property
    def checkpoint_ready(self) -> bool:
        """True when a complete checkpoint exists. SaveQudaEigenPack writes
        <stem>.done only after every rank has finished writing, so a write
        interrupted by a crash is never mistaken for a good one."""
        return self.checkpoint and os.path.exists(self.checkpoint_stem + ".done")

    @property
    def solve(self) -> bool:
        """True when this job runs the IRL (rather than loading eigenvectors)."""
        return not (self.load or self.checkpoint_ready)

    @property
    def masses(self) -> t.List[str]:
        return ["zero"] if self.solve else []


def build_input_params(config: EpackConfig) -> HadronsInput:
    modules = {}
    schedule = []
    multifile = str(config.multifile).lower()
    epack_path = ""
    if config.load or config.save_eigs:
        epack_path = config.eig.filestem

    # Load or generate eigenvectors
    if config.checkpoint_ready:
        utils.get_logger().info(
            f"Reusing checkpointed eigenpack {config.checkpoint_stem}"
        )
        modules["epack"] = hadmods.quda_epack_read(
            name="epack",
            filestem=config.checkpoint_stem,
            size=str(config.eigs),
        )
    elif config.load:
        modules["epack"] = hadmods.epack_load(
            name="epack",
            filestem=epack_path,
            size=str(config.eigs),
            multifile=multifile,
        )
    else:
        modules["stag_op"] = hadmods.op(
            "stag_op", config.action_name.format(mass="zero")
        )
        schedule.append("stag_op")
        modules["epack"] = hadmods.irl(
            name="epack",
            op="stag_op_schur",
            multifile=multifile,
            output=epack_path,
            **config.lanczos.to_string(),
        )
    schedule.append("epack")

    # Save the checkpoint immediately after the IRL, before anything else runs.
    # mass=0: eigenvalues are stored as (0, sqrt(lambda)) and read back as
    # lambda exactly; ModifyEigenPackMILC applies the quark mass afterwards.
    if config.solve and config.checkpoint:
        # Grid's partfile writer does not create directories; make the
        # checkpoint directory now (input files are written before submission).
        os.makedirs(os.path.dirname(config.checkpoint_stem) or ".", exist_ok=True)
        modules["epack_save"] = hadmods.quda_epack_save(
            name="epack_save",
            eigen_pack="epack",
            mass="0.0",
            output=config.checkpoint_stem,
            single_prec=str(config.checkpoint_single_prec).lower(),
        )
        schedule.append("epack_save")

    if config.save_evals:
        eval_path = config.eval.filestem
        modules["eval_save"] = hadmods.eval_save(
            name="eval_save", eigen_pack="epack", output=eval_path
        )
        schedule.append("eval_save")
    return HadronsInput(modules=modules, schedule=schedule)


def build_epack_mass_shifts(
    config: EpackConfig,
    mass_shifts: t.List[str],
) -> HadronsInput:
    """Build epack mass shift modules for specified mass labels.

    Args:
        config: EpackConfig with mass definitions and low_modes_name template
        mass_shifts: List of mass labels to shift eigenvalues for

    Returns:
        HadronsInput with modules and schedule for mass shifts
    """
    modules = {}
    schedule = []

    for mass_label in mass_shifts:
        mass = config.mass.to_string(mass_label)
        name = config.low_modes_name.format(mass=mass_label)
        modules[name] = hadmods.epack_modify(name=name, eigen_pack="epack", mass=mass)
        schedule.append(name)

    return HadronsInput(modules=modules, schedule=schedule)


def create_outfile_catalog(config: EpackConfig) -> pd.DataFrame:
    def generate_outfile_formatting():
        """
        A generator function that yields file formatting details for different
        components of the task configuration.

        Yields:
            Tuple[Dict[str, Any], str]: A dictionary of replacements and the
            corresponding output file path for each component.
        """

        if config.save_eigs:
            if config.multifile:
                yield ({"eig_index": list(range(int(config.eigs)))}, config.eigdir)
            else:
                yield {}, config.eig
        if config.save_evals:
            yield {}, config.eval

    outfile_generator = generate_outfile_formatting()

    df = utils.io.catalog_files(outfile_generator)

    return df


def validate_config(config: EpackConfig) -> None:
    """Validate EpackConfig after construction and postprocessing.

    Validates:
    - If not loading, requires Lanczos parameters and action_name
    - If checkpointing, requires an eig_quda file entry
    """
    if config.checkpoint and config.eig_quda is None:
        raise ValueError(
            "epack checkpoint: true requires an 'eig_quda' entry under files:"
        )
    if config.checkpoint and config.load:
        raise ValueError("epack checkpoint and load are mutually exclusive")
    if not config.load:
        if config.lanczos is None:
            raise ValueError(
                "No Lanczos parameters provided. Required for using IRL solver."
            )
        if config.action_name is None:
            raise ValueError(
                "No action_name parameter provided. Required for using IRL solver."
            )


# Register EpackConfig with all handlers
register_task(
    "hadrons_epack",
    EpackConfig,
    build_input_params,
    create_outfile_catalog,
    validate=validate_config,
)
