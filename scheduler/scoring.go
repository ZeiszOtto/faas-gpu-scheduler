package main

import (
	"fmt"
	"log"
	"math"
	"strings"
)

func selectNode(cfg *Config, gpuDB *GPUDatabase, nodeGPUMap map[string]string) (string, error) {
	// 1) Query GPU metrics
	metrics, err := QueryGPUMetrics(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to query GPU metrics: %w", err)
	}

	// 2) Score each node
	var bestNode string
	var bestScore float64 = -1

	for hostname, metric := range metrics {
		dbModelName, exists := nodeGPUMap[hostname]
		if !exists {
			log.Printf("[WARNING] No GPU model mapping for node %s — skipping", hostname)
			continue
		}

		gpuSpec := gpuDB.GPUs[dbModelName]
		totalVRAMMB := gpuSpec.MemorySizeGB * 1024

		dynamicScore := computeDynamicScore(metric, totalVRAMMB)

		staticScore, err := gpuDB.GetGPUScore(dbModelName, cfg.ScoringPreset, cfg.TensorScoring)
		if err != nil {
			log.Printf("[ERROR] Could not compute score for %q: %v — skipping", dbModelName, err)
			continue
		}

		capWeight := cfg.CapabilityWeight
		finalScore := (1-capWeight)*dynamicScore + capWeight*staticScore

		log.Printf("[SCORING] %s: model=%s, dynamic=%.4f, static=%.4f, final=%.4f (capWeight=%.2f)",
			hostname, dbModelName, dynamicScore, staticScore, finalScore, capWeight)

		if finalScore > bestScore {
			bestScore = finalScore
			bestNode = hostname
		}
	}

	// 3) Return best node
	if bestNode == "" {
		return "", fmt.Errorf("no scoreable nodes found: all nodes failed GPU model matching or scoring")
	}
	log.Printf("[INFO] Selected node: %s (score=%.4f)", bestNode, bestScore)
	return bestNode, nil
}

func computeDynamicScore(metric NodeMetric, totalVRAM float64) float64 {
	idleRatio := 1.0 - (metric.GPUUtilization / 100.0)

	var vramRatio float64
	if totalVRAM > 0 {
		vramRatio = metric.VRAMFree / totalVRAM
	}

	idleRatio = clamp(idleRatio, 0.0, 1.0)
	vramRatio = clamp(vramRatio, 0.0, 1.0)

	return 0.5*idleRatio + 0.5*vramRatio
}

func matchGPUName(dcgmName string, gpuDB *GPUDatabase, totalVRAMMB float64) (string, bool) {
	if dcgmName == "" {
		return "", false
	}

	strippedName := strings.TrimPrefix(dcgmName, "NVIDIA ")

	// Try exact matching
	if _, exists := gpuDB.GPUs[strippedName]; exists {
		return strippedName, true
	}

	// Fallback to prefix matching
	var matches []string
	for dbName := range gpuDB.GPUs {
		if strings.HasPrefix(dbName, strippedName) {
			matches = append(matches, dbName)
		}
	}

	switch len(matches) {
	case 0:
		return "", false
	case 1:
		return matches[0], true
	default:
		best := matches[0]
		bestDiff := math.Abs(gpuDB.GPUs[best].MemorySizeGB*1024 - totalVRAMMB)

		for _, m := range matches[1:] {
			diff := math.Abs(gpuDB.GPUs[m].MemorySizeGB*1024 - totalVRAMMB)
			if diff < bestDiff {
				best = m
				bestDiff = diff
			}
		}
		log.Printf("[WARNING] Multiple GPU database matches for %q: %v — selected %q (by VRAM)",
			dcgmName, matches, best)
		return best, true
	}
}

// clamp restricts a value to the [min, max] range.
func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
