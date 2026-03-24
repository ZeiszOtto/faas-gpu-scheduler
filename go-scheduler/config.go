package main

import (
	"log"
	"os"
	"strconv"
	"time"
)

type Config struct {
	TLSCertFile     string
	TLSKeyFile      string
	Port            string
	PrometheusURL   string
	MetricWindow    time.Duration
	GPUUtilWeight   float64
	VRAMWeight      float64
	TargetNamespace string
}

// LoadConfig returns ...
func LoadConfig() *Config {
	cfg := &Config{
		TLSCertFile:     getEnv("TLS_CERT_FILE", "etc/webhook/tls/tls.crt"),
		TLSKeyFile:      getEnv("TLS_KEY_FILE", "etc/webhooko/tls/tls.key"),
		Port:            getEnv("PORT", "8443"),
		PrometheusURL:   getEnv("PROMETHEUS_URL", "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"),
		TargetNamespace: getEnv("TARGET_NAMESPACE", "default"),
	}

	// MetricWindow: string -> time.Duration conversion
	windowStr := getEnv("METRIC_WINDOW", "30s")
	window, err := time.ParseDuration(windowStr)
	if err != nil {
		log.Fatalf("Invalid value for METRIC_WINDOW: %q", windowStr)
	}
	cfg.MetricWindow = window

	// GPUUtilWeight: string -> float64 conversion
	gpuWeightStr := getEnv("GPU_WEIGHT", "0.6")
	gpuWeight, err := strconv.ParseFloat(gpuWeightStr, 64)
	if err != nil {
		log.Fatalf("Invalid value for GPU_WEIGHT: %q", gpuWeightStr)
	}
	cfg.GPUUtilWeight = gpuWeight

	// VRAMWeight: string -> float64 conversion
	vramWeightStr := getEnv("VRAM_WEIGHT", "0.4")
	vramWeight, err := strconv.ParseFloat(vramWeightStr, 64)
	if err != nil {
		log.Fatalf("Invalid value for VRAM_WEIGHT: %q", vramWeightStr)
	}
	cfg.VRAMWeight = vramWeight

	// Weight validation
	sum := gpuWeight + vramWeight
	if sum < 0.99 || sum > 1.01 {
		log.Fatalf("Weight is %.2f, but it needs to be 1.00", sum)
	}

	log.Printf("Configuration loaded: port=%s, prometheus=%s, window=%s, gpuWeight=%.2f, vramWeight=%.2f",
		cfg.Port, cfg.PrometheusURL, cfg.MetricWindow, cfg.GPUUtilWeight, cfg.VRAMWeight)

	return cfg
}

// getEnv returns the requested environment variable. If the requested variable is not present, it returns the set default value.
func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}
