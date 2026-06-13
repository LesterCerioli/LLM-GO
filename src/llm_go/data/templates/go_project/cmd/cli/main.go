// cmd/cli/main.go — Cobra CLI entry point.
// cmd/ is ALWAYS at the project root. main() delegates entirely to internal/cmd.
package main

import (
	"os"

	"github.com/example/myapp/internal/cmd"
)

func main() {
	if err := cmd.Execute(); err != nil {
		os.Exit(1)
	}
}
