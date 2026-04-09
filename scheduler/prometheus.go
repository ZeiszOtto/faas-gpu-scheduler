package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// NodeMetric contains the GPU Utilization and remaining VRAM values for a node
type NodeMetric struct {
	GPUUtilization float64
	VRAMFree       float64
	GPUModel       string
}

// prometheusResponse contains the outer section of the Prometheus API response that contains
// the HTTP status code and the data section that contains the inner sections of the response.
type prometheusResponse struct {
	Status string         `json:"status"` // "status": "success" | "error"
	Data   prometheusData `json:"data"`
}

// prometheusData contains the middle section of the Prometheus API response that contains
// the structural identifier of the result and the result itself.
type prometheusData struct {
	ResultType string             `json:"resultType"` // "resultType": "matrix" | "vector" | "scalar" | "string"
	Result     []prometheusResult `json:"result"`
}

// prometheusResult is the inner layer of the Prometheus API response that contains the metric
// labels (including Hostname to identify the node) and the measured value as a two-element array.
type prometheusResult struct {
	Metric map[string]string `json:"metric"`
	Value  []interface{}     `json:"value"` // "value": [ 1435781451.781, "1" ]
}

// QueryGPUMetrics queries Prometheus for GPU utilization and free VRAM metrics across all nodes
// using PromQL with moving average smoothing. Returns a map of node hostname to NodeMetric.
// If Prometheus is unreachable or returns no results the webhook falls back to allowing the pod without patching.
func QueryGPUMetrics(cfg *Config) (map[string]NodeMetric, error) {
	// Turning the window duration into a string for PromQL
	window := fmt.Sprintf("%ds", int(cfg.MetricWindow.Seconds()))

	// Creating the PromQL queries with avg_over_time to avoid momentary GPU spikes poisoning the query
	queryUtil := fmt.Sprintf("avg_over_time(DCGM_FI_DEV_GPU_UTIL[%s])", window)
	queryVRAM := fmt.Sprintf("avg_over_time(DCGM_FI_DEV_FB_FREE[%s])", window)

	// Querying
	utilizationResults, err := queryPrometheus(cfg.PrometheusURL, queryUtil)
	if err != nil {
		return nil, fmt.Errorf("error querying GPU Metrics: %w", err)
	}
	vramResults, err := queryPrometheus(cfg.PrometheusURL, queryVRAM)
	if err != nil {
		return nil, fmt.Errorf("error querying VRAM Metrics: %w", err)
	}

	// Extracting the queried metrics of each node and adding them to a map where the key is the hostname
	// and the value is the NodeMetric struct that contains the GPU Utilization and VRAM Usage values
	metrics := make(map[string]NodeMetric)

	for _, result := range utilizationResults {
		hostname, value, err := extractHostnameAndValue(result)
		if err != nil {
			log.Printf("[ERROR] An error occurred during extracting utilization metrics: %v", err)
			continue
		}
		entry := metrics[hostname]   // entry = NodeMetrics{GPUUtil: 0.0, VRAMFree: 0.0}
		entry.GPUUtilization = value // entry = NodeMetrics{GPUUtil: 45.2, VRAMFree: 0.0}

		if modelName, exists := result.Metric["modelName"]; exists {
			entry.GPUModel = modelName
		}

		metrics[hostname] = entry // metrics = {"rtx-master": NodeMetrics{GPUUtil: 45.2, VRAMFree: 0.0}}
	}

	for _, result := range vramResults {
		hostname, value, err := extractHostnameAndValue(result)
		if err != nil {
			log.Printf("[ERROR] An error occurred during extracting VRAM metrics: %v", err)
			continue
		}
		entry := metrics[hostname]
		entry.VRAMFree = value
		metrics[hostname] = entry
	}

	// Logging gathered metrics
	for node, m := range metrics {
		log.Printf("[INFO/METRICS] %s: GPU=%.1f%% | VRAM_FREE=%.0f MB | MODEL=%s",
			node, m.GPUUtilization, m.VRAMFree, m.GPUModel)
	}

	if len(metrics) == 0 {
		return nil, fmt.Errorf("no DCGM metrics found")
	}

	return metrics, nil
}

// queryPrometheus sends a specified query request towards the Prometheus server and processes the
// response into a prometheusResult struct that contains the metrics and their values.
func queryPrometheus(prometheusURL string, query string) ([]prometheusResult, error) {
	// Assembling URL with QueryEscape to encode special characters into PromQL
	fullURL := fmt.Sprintf("%s/api/v1/query?query=%s", prometheusURL, url.QueryEscape(query))

	// HTTP Client with 3 second timeout configured
	client := &http.Client{Timeout: 3 * time.Second}

	response, err := client.Get(fullURL)
	if err != nil {
		return nil, fmt.Errorf("unsuccessful Prometheus query: %w", err)
	}
	defer response.Body.Close()

	// Status checking
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unsuccessful Prometheus query: %v", response.Status)
	}

	// Reading response body
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("error reading response body: %w", err)
	}

	// JSON parsing
	var promResponse prometheusResponse
	if err := json.Unmarshal(body, &promResponse); err != nil {
		return nil, fmt.Errorf("error parsing JSON: %w", err)
	}

	if promResponse.Status != "success" {
		return nil, fmt.Errorf("prometheus status: %s", promResponse.Status)
	}

	if promResponse.Data.ResultType != "vector" {
		return nil, fmt.Errorf("vector ResultType expected, instead: %s", promResponse.Data.ResultType)
	}

	return promResponse.Data.Result, nil
}

// BuildNodeGPUMap queries DCGM metrics once at startup to build a mapping of node hostnames
// to GPU database model names.
func BuildNodeGPUMap(cfg *Config, gpuDB *GPUDatabase) (map[string]string, error) {
	// Query total VRAM per node: free + used = constant regardless of load
	query := "DCGM_FI_DEV_FB_FREE + DCGM_FI_DEV_FB_USED"
	results, err := queryPrometheus(cfg.PrometheusURL, query)
	if err != nil {
		return nil, fmt.Errorf("failed to query VRAM for GPU model resolution: %w", err)
	}

	nodeGPUMap := make(map[string]string)

	for _, result := range results {
		hostname, totalVRAM, err := extractHostnameAndValue(result)
		if err != nil {
			log.Printf("[ERROR] Failed to extract VRAM data: %v", err)
			continue
		}

		modelName, exists := result.Metric["modelName"]
		if !exists {
			log.Printf("[WARNING] No modelName label found for node %s", hostname)
			continue
		}

		dbName, found := matchGPUName(modelName, gpuDB, totalVRAM)
		if !found {
			log.Printf("[WARNING] No GPU database match for %q on node %s", modelName, hostname)
			continue
		}

		nodeGPUMap[hostname] = dbName
		log.Printf("[INFO/GPUMAP] %s → %s (DCGM: %q, totalVRAM: %.0f MB)", hostname, dbName, modelName, totalVRAM)
	}

	if len(nodeGPUMap) == 0 {
		return nil, fmt.Errorf("no GPU models could be resolved from DCGM metrics")
	}

	return nodeGPUMap, nil
}

// extractHostnameAndValue extracts the hostname of the node and the requested value from the prometheusResponse
func extractHostnameAndValue(result prometheusResult) (string, float64, error) {
	hostname, exists := result.Metric["Hostname"]
	if !exists {
		return "", 0, fmt.Errorf("hostname not found in result")
	}

	if len(result.Value) < 2 {
		return "", 0, fmt.Errorf("invalid value array on %s node", hostname)
	}

	// Discard the UNIX timestamp and only keep the value
	valueStr, ok := result.Value[1].(string)
	if !ok {
		return "", 0, fmt.Errorf("value on %s node couldn't be converted", hostname)
	}

	value, err := strconv.ParseFloat(valueStr, 64)
	if err != nil {
		return "", 0, fmt.Errorf("error parsing value on %s node: %w", hostname, err)
	}

	return hostname, value, nil
}
