package main

import (
	"log"
	"os"
	"strconv"
	"time"
)

type Config struct {
	TLSCertFile      string
	TLSKeyFile       string
	Port             string
	PrometheusURL    string
	MetricWindow     time.Duration
	TargetNamespace  string
	GPUDatabasePath  string
	ScoringPreset    string
	TensorScoring    bool
	CapabilityWeight float64
}

// LoadConfig reads all configuration from environment variables, performs type conversions and validation,
// and returns a fully populated Config pointer. Called once at startup from main(). Configuration errors are fatal.
func LoadConfig() *Config {
	cfg := &Config{
		TLSCertFile:     getEnv("TLS_CERT_FILE", "/etc/webhook/tls/tls.crt"),
		TLSKeyFile:      getEnv("TLS_KEY_FILE", "/etc/webhook/tls/tls.key"),
		Port:            getEnv("PORT", "8443"),
		PrometheusURL:   getEnv("PROMETHEUS_URL", "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090"),
		TargetNamespace: getEnv("TARGET_NAMESPACE", "default"),
		GPUDatabasePath: getEnv("GPU_DATABASE_PATH", "/etc/webhook/gpu_database.json"),
		ScoringPreset:   getEnv("SCORE_PRESET", "inference"),
	}

	// MetricWindow: string -> time.Duration conversion
	windowStr := getEnv("METRIC_WINDOW", "30s")
	window, err := time.ParseDuration(windowStr)
	if err != nil {
		log.Fatalf("Invalid value for METRIC_WINDOW: %q", windowStr)
	}
	cfg.MetricWindow = window

	// TensorScoring type conversion
	tensorStr := getEnv("TENSOR_SCORING", "true")
	tensorScoring, err := strconv.ParseBool(tensorStr)
	if err != nil {
		log.Fatalf("Invalid value for TENSOR_SCORING: %q", tensorStr)
	}
	cfg.TensorScoring = tensorScoring

	// CapabilityWeight: string -> float64 conversion
	capWeightStr := getEnv("CAPABILITY_WEIGHT", "0.2")
	capWeight, err := strconv.ParseFloat(capWeightStr, 64)
	if err != nil {
		log.Fatalf("Invalid value for CAPABILITY_WEIGHT: %q", capWeightStr)
	}
	if capWeight < 0.0 || capWeight > 1.0 {
		log.Fatalf("CAPABILITY_WEIGHT must be between 0.0 and 1.0, got: %.2f", capWeight)
	}
	cfg.CapabilityWeight = capWeight

	log.Printf("[INFO] Configuration loaded: port=%s, prometheus=%s, window=%s",
		cfg.Port, cfg.PrometheusURL, cfg.MetricWindow)

	return cfg
}

// getEnv returns the requested environment variable. If the requested variable is not present, it returns the set default value.
func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}
