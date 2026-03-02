"""
Project: top-level container for a pipeline of modules.

Handles:
- Module registration and dependency ordering (topological sort)
- Running modules in order, passing Context between them
- Dry-run mode (modules see ctx.dry_run=True and skip actual submission)
"""

from __future__ import annotations

from zoo.module import Context, Module, ModuleResult, ModuleStatus


class Project:
    """
    Root container for a set of modules.

    Usage::

        project = Project(name="qwen3-moe-ablations", cluster="magnum")
        project.add(Sweep(name="find-best-tp", ...))
        project.add(ExperimentChain(name="qwen3-33b-run", depends_on=["find-best-tp"], ...))
        project.run()
    """

    def __init__(self, name: str, cluster: str | None = None):
        self.name = name
        self.cluster = cluster
        self._modules: dict[str, Module] = {}
        self._order: list[str] = []  # insertion order before topo-sort

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, module: Module) -> "Project":
        """Register a module. Returns self for chaining."""
        if module.name in self._modules:
            raise ValueError(f"Module '{module.name}' already registered in project '{self.name}'")
        self._modules[module.name] = module
        self._order.append(module.name)
        return self

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> dict[str, ModuleResult]:
        """
        Run all modules in dependency order.

        Returns a dict of module_name → ModuleResult.
        """
        ctx = Context(project_name=self.name, dry_run=dry_run)
        run_order = self._topo_sort()

        for name in run_order:
            module = self._modules[name]

            # Check all dependencies completed successfully
            for dep in module.depends_on:
                if dep not in ctx:
                    raise RuntimeError(
                        f"Module '{name}' depends on '{dep}' but it has not run yet."
                    )
                dep_result = ctx[dep]
                if dep_result.status not in (ModuleStatus.DONE, ModuleStatus.SKIPPED):
                    print(
                        f"[project] Skipping '{name}': dependency '{dep}' "
                        f"finished with status {dep_result.status.value}"
                    )
                    ctx.record(ModuleResult(
                        module_name=name,
                        status=ModuleStatus.SKIPPED,
                        error=f"Dependency '{dep}' did not complete successfully",
                    ))
                    continue

            print(f"[project] Running '{name}' ...")
            result = module.run(ctx)
            ctx.record(result)

            if result.status == ModuleStatus.FAILED:
                print(f"[project] Module '{name}' FAILED: {result.error}")
            else:
                print(f"[project] Module '{name}' → {result.status.value}")

        return ctx.all_results()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_modules(self) -> list[str]:
        """Return module names in dependency-sorted order."""
        return self._topo_sort()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _topo_sort(self) -> list[str]:
        """
        Kahn's algorithm topological sort over module depends_on edges.
        Preserves insertion order for independent modules.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._order}
        dependents: dict[str, list[str]] = {n: [] for n in self._order}

        for name in self._order:
            for dep in self._modules[name].depends_on:
                if dep not in self._modules:
                    raise ValueError(
                        f"Module '{name}' depends on '{dep}' which is not registered."
                    )
                in_degree[name] += 1
                dependents[dep].append(name)

        # Start with all zero-in-degree nodes, preserving insertion order
        queue = [n for n in self._order if in_degree[n] == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for child in dependents[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self._order):
            cycle_nodes = set(self._order) - set(result)
            raise ValueError(f"Dependency cycle detected among modules: {cycle_nodes}")

        return result
