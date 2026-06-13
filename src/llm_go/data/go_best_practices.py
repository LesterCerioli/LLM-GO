"""
Go project layout conventions and best practices injected into training data.

The central rule enforced here:

    When a Go project has a main application, the entry point MUST live at
    <project-root>/cmd/<app-name>/main.go

    cmd/ is ALWAYS at the repository root — never nested inside internal/,
    pkg/, or any other directory.

This module:
  1. Defines the canonical Go project layout as structured text
  2. Provides a template renderer for synthetic training examples
  3. Verifies that scraped repos follow the convention (soft signal for data quality)
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from string import Template


# ---------------------------------------------------------------------------
# Canonical layout spec (used verbatim in training prompts and docs)
# ---------------------------------------------------------------------------

GO_PROJECT_LAYOUT = """\
<go_file>
// Go standard project layout — embedded as training reference.
// Rule: cmd/ MUST be at the project root. Each sub-directory is one binary
// and MUST contain a main.go that calls into internal packages.
//
// my-project/             ← repository root
// ├── cmd/                ← ALWAYS at root; never nested
// │   ├── server/
// │   │   └── main.go     ← entry point for the 'server' binary
// │   └── worker/
// │       └── main.go     ← entry point for the 'worker' binary
// ├── internal/           ← private packages (not importable by others)
// │   ├── config/
// │   ├── handler/
// │   ├── service/
// │   └── repository/
// ├── pkg/                ← public, reusable packages
// ├── api/                ← protobuf / OpenAPI definitions
// ├── web/                ← static assets / templates
// ├── scripts/            ← build / CI scripts
// ├── docs/
// ├── go.mod
// └── go.sum
package layout
</go_file>
"""

# ---------------------------------------------------------------------------
# Synthetic training examples
# ---------------------------------------------------------------------------

_CMD_MAIN_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> main

// cmd/$app_name/main.go — entry point for the $app_name binary.
// This file lives at the repository root under cmd/$app_name/.
// All business logic belongs in internal/ or pkg/, not here.
package main

import (
\t"context"
\t"log/slog"
\t"os"
\t"os/signal"
\t"syscall"

\t"$module/internal/config"
\t"$module/internal/$app_name"
)

func main() {
\tlogger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
\t\tLevel: slog.LevelInfo,
\t}))
\tslog.SetDefault(logger)

\tcfg, err := config.Load()
\tif err != nil {
\t\tlogger.Error("failed to load config", "err", err)
\t\tos.Exit(1)
\t}

\tctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
\tdefer stop()

\tapp := ${app_name_cap}.New(cfg, logger)
\tif err := app.Run(ctx); err != nil {
\t\tlogger.Error("application error", "err", err)
\t\tos.Exit(1)
\t}
}
</go_file>
""")

_GOMOD_TEMPLATE = Template("""\
<go_file>
// go.mod — module declaration at the repository root.
module $module

go $go_version

require (
\tgithub.com/gofiber/fiber/v2 v2.52.4
\tgithub.com/spf13/cobra v1.8.0
\tgithub.com/spf13/viper v1.18.2
\tgo.uber.org/zap v1.27.0
)
</go_file>
""")

_INTERNAL_SERVICE_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> $app_name

// internal/$app_name/$app_name.go
// Business logic lives here, NOT in cmd/.
// cmd/$app_name/main.go wires dependencies and calls Run().
package ${app_name}

import (
\t"context"
\t"fmt"
\t"log/slog"

\t"$module/internal/config"
)

// App is the root application struct.
type App struct {
\tcfg    *config.Config
\tlogger *slog.Logger
}

// New constructs App with all dependencies injected.
func New(cfg *config.Config, logger *slog.Logger) *App {
\treturn &App{cfg: cfg, logger: logger}
}

// Run starts the application and blocks until ctx is cancelled.
func (a *App) Run(ctx context.Context) error {
\ta.logger.Info("starting", "app", "$app_name", "version", a.cfg.Version)
\t<-ctx.Done()
\ta.logger.Info("shutting down gracefully")
\treturn nil
}
</go_file>
""")

_FIBER_SERVER_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> main

// cmd/server/main.go — Fiber HTTP server entry point.
// cmd/ is at the project root. main() only wires dependencies.
package main

import (
\t"log/slog"
\t"os"

\t"github.com/gofiber/fiber/v2"
\t"github.com/gofiber/fiber/v2/middleware/logger"
\t"github.com/gofiber/fiber/v2/middleware/recover"

\t"$module/internal/config"
\t"$module/internal/handler"
)

func main() {
\tlog := slog.New(slog.NewJSONHandler(os.Stdout, nil))

\tcfg, err := config.Load()
\tif err != nil {
\t\tlog.Error("config error", "err", err)
\t\tos.Exit(1)
\t}

\tapp := fiber.New(fiber.Config{
\t\tAppName:       "$app_name",
\t\tReadTimeout:   cfg.ReadTimeout,
\t\tWriteTimeout:  cfg.WriteTimeout,
\t\tErrorHandler:  handler.ErrorHandler,
\t})

\tapp.Use(recover.New())
\tapp.Use(logger.New(logger.Config{
\t\tFormat: `{"time":"$${time}","status":$${status},"method":"$${method}","path":"$${path}"}` + "\\n",
\t}))

\th := handler.New(cfg, log)
\th.Register(app)

\tlog.Info("listening", "addr", cfg.Addr)
\tif err := app.Listen(cfg.Addr); err != nil {
\t\tlog.Error("server error", "err", err)
\t\tos.Exit(1)
\t}
}
</go_file>
""")

_COBRA_MAIN_TEMPLATE = Template("""\
<go_file>
<go_version> go$go_version
<go_pkg> main

// cmd/$app_name/main.go — Cobra CLI entry point.
// Rule: cmd/ is always at the project root.
// All command logic lives in internal/cmd/, not here.
package main

import (
\t"os"

\t"$module/internal/cmd"
)

func main() {
\tif err := cmd.Execute(); err != nil {
\t\tos.Exit(1)
\t}
}
</go_file>
""")


# ---------------------------------------------------------------------------
# Template renderer
# ---------------------------------------------------------------------------

class GoProjectTemplates:
    """
    Renders synthetic Go project examples for injection into training data.

    Every example enforces the cmd/ layout rule:
        <project-root>/cmd/<app>/main.go
    """

    GO_VERSIONS = [
        "1.16", "1.17", "1.18", "1.19", "1.20",
        "1.21", "1.22", "1.23", "1.24",
    ]

    APPS = [
        ("api",      "Api"),
        ("server",   "Server"),
        ("worker",   "Worker"),
        ("cli",      "Cli"),
        ("gateway",  "Gateway"),
        ("migrator", "Migrator"),
        ("scraper",  "Scraper"),
        ("bot",      "Bot"),
    ]

    def __init__(self, module: str = "github.com/example/myapp"):
        self.module = module

    def all_examples(self) -> list[str]:
        """Return all synthetic training examples as strings."""
        examples: list[str] = [GO_PROJECT_LAYOUT]

        for go_ver in self.GO_VERSIONS:
            for app_name, app_cap in self.APPS:
                examples += [
                    self._cmd_main(go_ver, app_name, app_cap),
                    self._internal_service(go_ver, app_name),
                ]
            examples.append(self._gomod(go_ver))

        # Framework-specific examples
        for go_ver in ["1.21", "1.22", "1.23", "1.24"]:
            examples.append(self._fiber_server(go_ver, "api"))
            examples.append(self._cobra_main(go_ver, "cli"))

        return examples

    def _cmd_main(self, ver: str, app: str, cap: str) -> str:
        return _CMD_MAIN_TEMPLATE.substitute(
            go_version=ver, app_name=app, app_name_cap=cap, module=self.module
        )

    def _gomod(self, ver: str) -> str:
        return _GOMOD_TEMPLATE.substitute(go_version=ver, module=self.module)

    def _internal_service(self, ver: str, app: str) -> str:
        return _INTERNAL_SERVICE_TEMPLATE.substitute(
            go_version=ver, app_name=app, module=self.module
        )

    def _fiber_server(self, ver: str, app: str) -> str:
        return _FIBER_SERVER_TEMPLATE.substitute(
            go_version=ver, app_name=app, module=self.module
        )

    def _cobra_main(self, ver: str, app: str) -> str:
        return _COBRA_MAIN_TEMPLATE.substitute(
            go_version=ver, app_name=app, module=self.module
        )


# ---------------------------------------------------------------------------
# Layout validator (used during data collection quality filtering)
# ---------------------------------------------------------------------------

class GoLayoutValidator:
    """
    Validates that a scraped repo follows the cmd/ layout convention.
    Used as a soft quality signal — does NOT discard files, only annotates.
    """

    @staticmethod
    def validate(file_paths: list[str]) -> dict[str, bool]:
        """
        Given a list of repo file paths, return layout quality signals.

        Returns:
            has_cmd_at_root    — True if cmd/ exists at root level
            cmd_has_main_go    — True if at least one cmd/**/main.go exists
            no_nested_cmd      — True if cmd/ does NOT appear below root
        """
        has_cmd_at_root = any(
            p.startswith("cmd/") for p in file_paths
        )
        cmd_has_main_go = any(
            p.startswith("cmd/") and p.endswith("/main.go") for p in file_paths
        )
        # Nested cmd would be something like internal/cmd/foo/bar/ — flag it
        no_nested_cmd = not any(
            "/" in p and "cmd/" in p and not p.startswith("cmd/")
            for p in file_paths
        )
        return {
            "has_cmd_at_root": has_cmd_at_root,
            "cmd_has_main_go": cmd_has_main_go,
            "no_nested_cmd":   no_nested_cmd,
        }

    @staticmethod
    def layout_score(validation: dict[str, bool]) -> float:
        """0.0–1.0 quality score for weighting during training."""
        return sum(validation.values()) / len(validation)
