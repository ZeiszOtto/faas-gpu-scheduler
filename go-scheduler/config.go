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

/*
...
*/
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
		log.Fatalf("Error parsing METRIC_WINDOW: %v", err)
	}
	cfg.MetricWindow = window

	//
	gpuWeightStr := getEnv("GPU_WEIGHT", "0.6")
	gpuWeight, err := strconv.ParseFloat(gpuWeightStr, 64)

}

/*
Returns the requested environment variable.
If the requested variable is not present, it returns the set default value.
*/
func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}
