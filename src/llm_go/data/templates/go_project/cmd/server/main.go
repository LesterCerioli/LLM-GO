// cmd/server/main.go — canonical entry point for the 'server' binary.
//
// LAYOUT RULE: cmd/ is ALWAYS at the project root.
// This file only wires dependencies; all business logic is in internal/.
//
//	my-project/
//	├── cmd/
//	│   └── server/
//	│       └── main.go   ← you are here
//	├── internal/
//	│   ├── config/
//	│   ├── handler/
//	│   └── service/
//	├── go.mod
//	└── go.sum
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"

	"github.com/example/myapp/internal/config"
	"github.com/example/myapp/internal/handler"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(log)

	cfg, err := config.Load()
	if err != nil {
		log.Error("failed to load configuration", "err", err)
		os.Exit(1)
	}

	app := fiber.New(fiber.Config{
		AppName:      cfg.AppName,
		ReadTimeout:  cfg.ReadTimeout,
		WriteTimeout: cfg.WriteTimeout,
		ErrorHandler: handler.ErrorHandler,
	})

	app.Use(recover.New())
	app.Use(logger.New())

	h := handler.New(cfg, log)
	h.Register(app)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("server listening", "addr", cfg.Addr)
		if err := app.Listen(cfg.Addr); err != nil {
			log.Error("server error", "err", err)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down gracefully…")
	if err := app.ShutdownWithContext(ctx); err != nil {
		log.Error("shutdown error", "err", err)
	}
}
