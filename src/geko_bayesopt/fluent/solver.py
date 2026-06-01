"""
Solver: 2D RANS / GEKO simulation for the periodic-hill case.

Two usage patterns:

1. **One-shot run** (single case, launch + setup + iterate + exit):

       PeriodicHillSolver(case, mesh_path, data_dir).run()

2. **Persistent session** (Fluent stays alive across multiple trials,
   useful for Bayesian optimization over GEKO coefficients):

       with PeriodicHillSolver(base_case, mesh_path, data_dir) as session:
           for trial_case in trial_cases:
               outputs = session.run_trial(trial_case)
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import time
import ansys.fluent.core as pyfluent

from .case_config import CaseConfig


class PeriodicHillSolver:
    """Run 2D RANS / GEKO simulations against a pre-generated mesh.

    Parameters
    ----------
    case : CaseConfig
        Initial / baseline case. For persistent sessions, only the parts
        that don't change between trials are read at construction (mesh
        location, zone names, iter_count default). GEKO and physics get
        re-applied per trial.
    mesh_path : str or Path
        Path to a .msh.h5 file produced by ``MeshGenerator``.
    data_dir : str or Path
        Where to write case/data/ASCII output files.
    ui_mode : str, optional
        PyFluent launch mode. Defaults to ``"no_gui_or_graphics"``.
    """

    # Variables exported to ASCII. Order doesn't matter for the extractor
    # (it reads by column name from the header). Keep this list in sync
    # with ``fluent.extract._FLUENT_COLUMNS`` if you add new fields.
    EXPORT_VARIABLES = [
        "pressure",
        "y-velocity",
        "x-velocity",
        "turb-kinetic-energy", #k
        "production-of-k",
    ]
    #     "k",
    #     "omega",
    #     "vorticity-mag",
    #     "wall-shear-stress",
    # ]

    def __init__(
        self,
        case: CaseConfig,
        mesh_path: str | Path,
        data_dir: str | Path,
        ui_mode: str = "no_gui_or_graphics",
        container_dict: dict | None = None,
        flow_case = None,
        residual_criteria: dict[str, float] | None = None,
    ):
        """Create a solver bound to one mesh and one base case.

        If ``flow_case`` is provided, its ``apply_boundary_conditions``
        method is called during ``start()`` instead of the hardcoded
        periodic-hill setup. This is how non-periodic flow cases (e.g.
        forward-facing step) plug in.

        Leaving ``flow_case=None`` preserves the original behaviour: the
        solver assumes periodic-hill BCs and configures them itself.
        """
        self.case = case
        self.mesh_path = Path(mesh_path).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.ui_mode = ui_mode
        self.container_dict = container_dict
        self.flow_case = flow_case
        self.residual_criteria = residual_criteria

        # Live Fluent session (None until start() is called)
        self._solver = None
        self._periodic_done = False

    # =========================================================================
    # ONE-SHOT USE
    # =========================================================================

    def run(self) -> dict[str, Path]:
        """Launch Fluent, run a single case, exit.

        Use this for one-off runs. For BO loops use ``start`` + ``run_trial``
        + ``close`` (or the context-manager form).
        """
        self.start()
        try:
            return self.run_trial(self.case)
        finally:
            self.close()

    # =========================================================================
    # PERSISTENT-SESSION USE
    # =========================================================================

    def start(self) -> None:
        """Launch Fluent and apply all setup that doesn't change between trials.

        Done once at the beginning of a BO loop. Sets up:
            - Solver dimensionality (2D planar, steady, pressure-based)
            - Periodic interface + mass-flow forcing
            - Turbulence model (k-omega GEKO selection only)
            - Material (constant density / viscosity)
            - Operating conditions
            - Pressure-velocity coupling

        GEKO coefficients are NOT applied here -- they go in run_trial
        because they're what BO sweeps over.
        """
        if self._solver is not None:
            raise RuntimeError("Solver session already started; call close() first.")

        self._validate_inputs()
        self._solver = self._launch()
        self._load_mesh()
        self._setup_solver_general()
        self._setup_boundary_conditions()
        self._setup_turbulence_model()
        self._setup_material()
        self._setup_operating_conditions()
        self._setup_methods()
        self._setup_residual_monitors()
        print("[solver] Session started and base case configured.")

    def run_trial(
        self,
        trial_case: CaseConfig,
        *,
        reinitialize: bool = True,
    ) -> dict[str, Path]:
        """Run one trial against the live Fluent session.

        Parameters
        ----------
        trial_case : CaseConfig
            The case to run this trial. GEKO coefficients are taken from
            this object; geometry/mesh/zone-name fields must be the same
            as the base case used in start().
        reinitialize : bool, default True
            Re-initialize the flow before iterating. Recommended True for
            BO so each trial is independent of the previous one's solution.
            Set False for warm-starting (e.g. continuation runs).

        Returns
        -------
        dict[str, Path]
            Output file paths, named by trial_case.case_id so multiple
            trials don't overwrite each other.
        """
        if self._solver is None:
            raise RuntimeError("Solver session not started; call start() first.")

        self._sanity_check_trial_case(trial_case)
        self._apply_geko_coefficients(trial_case)
        out = self._make_output_paths(trial_case)

        if reinitialize:
            self._solver.settings.solution.initialization.hybrid_initialize()

        # Save initial case (post-init, pre-iterate)
        self._solver.settings.file.write(
            file_name=str(out["case_init"]),
            file_type="case",
        )

        # Iterate
        self._solver.settings.solution.run_calculation.iterate(
            iter_count=trial_case.iter_count,
        )

        # Save converged
        self._solver.settings.file.write(
            file_name=str(out["case_solved"]),
            file_type="case",
        )

        

        self._solver.settings.file.write(
            file_name=str(out["data_solved"]),
            file_type="data",
        )

        # Export ASCII
        var_string = " ".join(self.EXPORT_VARIABLES) + " ()"
        ascii_path_str = Path(out["ascii"]).as_posix()
        self._solver.execute_tui(
            f'/file/export/ascii "{ascii_path_str}" () no {var_string} no'
        )

        print(f"[solver] Trial {trial_case.case_id} done. ASCII at {out['ascii']}")
        return out

    def close(self) -> None:
        """Exit Fluent and release the session."""
        if self._solver is not None:
            try:
                self._solver.exit()
            finally:
                self._solver = None
                self._periodic_done = False
                print("[solver] Session closed.")

    # =========================================================================
    # CONTEXT MANAGER -- ensures close() runs even on exceptions
    # =========================================================================

    def __enter__(self) -> "PeriodicHillSolver":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _validate_inputs(self) -> None:
        if self.container_dict is not None:
            return
        if not self.data_dir.is_dir():
            raise NotADirectoryError(f"data_dir does not exist: {self.data_dir!r}")
        if not self.mesh_path.is_file():
            raise FileNotFoundError(
                f"Mesh file not found: {self.mesh_path!r}. "
                "Run MeshGenerator first or pass an existing mesh."
            )

    def _sanity_check_trial_case(self, trial_case: CaseConfig) -> None:
        """Make sure the trial case isn't trying to change something that
        was baked in at start() time (geometry, zone names, etc.)."""
        immutable = (
            "geometry_basename", "alpha", "hill_height", "ly_over_h",
            "fluid_density", "fluid_viscosity", "re_h",
            "zone_inlet", "zone_outlet", "zone_top", "zone_bottom",
        )
        for field in immutable:
            if getattr(trial_case, field) != getattr(self.case, field):
                raise ValueError(
                    f"trial_case.{field} = {getattr(trial_case, field)!r} differs "
                    f"from base case ({getattr(self.case, field)!r}). "
                    f"This field is fixed at start() time; only GEKO coefficients "
                    f"and iter_count can vary between trials."
                )

    def _make_output_paths(self, trial_case: CaseConfig) -> dict[str, Path]:
        cid = trial_case.case_id
        return {
            "case_init": self.data_dir / f"{cid}_init.cas.h5",
            "case_solved": self.data_dir / f"{cid}_solved.cas.h5",
            "data_solved": self.data_dir / f"{cid}_solved.dat.h5",
            "ascii": self.data_dir / f"{cid}.ascii",
        }

    def _launch(self):
        kwargs = dict(
            precision="double",
            processor_count=4,
            mode="solver",
            dimension=2,
            cwd=str(self.data_dir),
        )
        if self.container_dict is not None:
            return pyfluent.launch_fluent(
                container_dict=self.container_dict,
                start_container=True,
                ui_mode="no_gui_or_graphics",
                cleanup_on_exit=False,
                start_timeout=300,
                **kwargs,
            )
        return pyfluent.launch_fluent(ui_mode=self.ui_mode, **kwargs)

    def _load_mesh(self) -> None:
        self._solver.settings.file.read_mesh(file_name=str(self.mesh_path))
        self._solver.settings.mesh.check()

    def _setup_solver_general(self) -> None:
        general = self._solver.settings.setup.general.solver
        general.two_dim_space = "planar"
        general.type = "pressure-based"
        general.time = "steady"

    def _setup_boundary_conditions(self) -> None:
        """Dispatch to the flow case's BC setup, or the periodic-hill default.

        Called once per session. If a ``flow_case`` was provided at
        construction, its ``apply_boundary_conditions`` runs. Otherwise
        the original periodic-hill setup runs (for backward compatibility
        with the standalone ``run.py`` script).
        """
        if self._periodic_done:
            return
        if self.flow_case is not None:
            self.flow_case.apply_boundary_conditions(self._solver)
        else:
            self._setup_periodic_default()
        self._periodic_done = True

    def _setup_periodic_default(self) -> None:
        """Original periodic-hill BC setup, kept for back-compat.

        Used only when no ``flow_case`` is provided.
        """
        self._solver.execute_tui(
            "/mesh/modify-zones/create-periodic-interface "
            "auto "
            f"{self.case.case_id} "
            f"{self.case.zone_inlet} "
            f"{self.case.zone_outlet} "
            "no "      # rotational? no = translational
            "yes "     # auto compute offset
            "yes "     # create periodic zones
        )
        self._solver.execute_tui(
            "/define/periodic-conditions/massflow-rate-specification? "
            f"{self.case.target_mass_flow} "
            "1 "        # initial pressure-gradient guess
            "0.5 "      # relaxation factor
            "1 "        # flow direction x
            "0 "        # flow direction y
        )

    def _setup_turbulence_model(self) -> None:
        """Select k-omega GEKO. Coefficients are applied per trial."""
        viscous = self._solver.settings.setup.models.viscous
        viscous.model = "k-omega"
        viscous.k_omega_model = "geko"

    def _apply_geko_coefficients(self, trial_case: CaseConfig) -> None:
        """Push the trial's GEKO coefficients to the live solver."""
        geko = self._solver.settings.setup.models.viscous.geko
        if trial_case.geko_csep is not None:
            geko.csep.value = trial_case.geko_csep
        if trial_case.geko_cnw is not None:
            geko.cnw.value = trial_case.geko_cnw
        if trial_case.geko_cmix is not None:
            geko.cmix.value = trial_case.geko_cmix
        if trial_case.geko_cjet is not None:
            geko.cjet.value = trial_case.geko_cjet
        if trial_case.geko_ccorner is not None:
            geko.ccorner.value = trial_case.geko_ccorner

    def _setup_material(self) -> None:
        air = self._solver.settings.setup.materials.fluid["air"]
        air.density.option = "constant"
        air.density.value = self.case.fluid_density
        air.viscosity.option = "constant"
        air.viscosity.value = self.case.fluid_viscosity

    def _setup_operating_conditions(self) -> None:
        try:
            self._solver.settings.setup.general.operating_conditions \
                .operating_pressure = 0.0
        except AttributeError:
            self._solver.execute_tui(
                "/define/operating-conditions/operating-pressure 0"
            )

    def _setup_methods(self) -> None:
        try:
            self._solver.settings.solution.methods.p_v_coupling.flow_scheme = "Coupled"
        except Exception:
            self._solver.tui.solve.set.p_v_coupling(24)

    def _setup_residual_monitors(self) -> None:
        """Set residual convergence criteria for Fluent.

        ``self.residual_criteria`` may be:
            - None: leave Fluent's defaults untouched
            - a Pydantic ``ResidualCriteria`` model
            - a plain dict with keys 'continuity', 'x-velocity', 'y-velocity',
              'k', 'omega' (or underscores -- both forms accepted)

        Fluent's TUI prompts for each residual in order, separated by
        newlines. We pass the values as space-separated arguments so the
        TUI reads them as successive responses.
        """
        if self.residual_criteria is None:
            return

        rc = self.residual_criteria
        # Accept either a Pydantic model (has attributes) or a dict.
        def _get(name_dash: str, name_under: str, default: float) -> float:
            if hasattr(rc, name_under):
                return float(getattr(rc, name_under))
            if isinstance(rc, dict):
                return float(rc.get(name_dash, rc.get(name_under, default)))
            return default

        continuity = _get("continuity", "continuity", 1.0e-3)
        x_velocity = _get("x-velocity", "x_velocity", 1.0e-3)
        y_velocity = _get("y-velocity", "y_velocity", 1.0e-3)
        k_val = _get("k", "k", 1.0e-3)
        omega_val = _get("omega", "omega", 1.0e-3)

        # Note the trailing spaces -- Fluent's TUI consumes each value
        # as a separate prompt response.
        self._solver.execute_tui(
            "/solve/monitors/residual/convergence-criteria "
            f"{continuity} "
            f"{x_velocity} "
            f"{y_velocity} "
            f"{k_val} "
            f"{omega_val} "
        )
        print(
            f"[solver] Residual criteria set: "
            f"cont={continuity}, u={x_velocity}, v={y_velocity}, "
            f"k={k_val}, omega={omega_val}"
        )
