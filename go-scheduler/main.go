package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	log.Println("[INFO] Starting scheduler...")

	cfg := LoadConfig()
	gpuDB, err := LoadGPUDatabase(cfg.GPUDatabasePath)
	if err != nil {
		log.Fatalf("[ERROR] Failed to load GPU database: %v", err)
	}

	if err := gpuDB.ValidatePreset(cfg.ScoringPreset); err != nil {
		log.Fatalf("[ERROR] Invalid scoring configuration: %v", err)
	}

	http.HandleFunc("/mutate", handleMutate(cfg, gpuDB))

	server := &http.Server{
		Addr:         ":" + cfg.Port,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
		IdleTimeout:  30 * time.Second,
	}

	// Goroutine of HTTPS server
	go func() {
		if err := server.ListenAndServeTLS(cfg.TLSCertFile, cfg.TLSKeyFile); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[ERROR] Unable to initiate HTTPS server on port %s: %v", cfg.Port, err)
		}
		log.Printf("[INFO] HTTPS server listening on port %s", cfg.Port)
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGTERM, syscall.SIGINT)
	sig := <-quit

	log.Printf("[INFO] Recieved signal %v - initiating shutdown...", sig)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("[ERROR] Unable to shutdown server: %v", err)
	}

	log.Printf("[INFO] Server stopped")
}
