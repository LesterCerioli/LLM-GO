// cmd/worker/main.go — entry point for the background worker binary.
// cmd/ is at the project root; this binary runs independently of the server.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/example/myapp/internal/config"
	"github.com/example/myapp/internal/worker"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	slog.SetDefault(log)

	cfg, err := config.Load()
	if err != nil {
		log.Error("config error", "err", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	w := worker.New(cfg, log)
	if err := w.Run(ctx); err != nil {
		log.Error("worker error", "err", err)
		os.Exit(1)
	}
}
