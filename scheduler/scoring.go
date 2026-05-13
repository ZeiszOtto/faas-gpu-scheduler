package main

import (
	"fmt"
	"log"
	"math"
	"strings"

	"k8s.io/client-go/kubernetes"
)

// selectNode is the core scheduling decision function. Depending on the configured PlacementStrategy
// it either runs the original single-objective capability scoring (capability_only) or the two-stage
// load_balanced strategy: first filter nodes to the ones with the lowest current replica count for the
// target service, then break the tie with capability scoring. Returns the chosen hostname, or an error
// if no node could be selected.
func selectNode(cfg *Config, gpuDB *GPUDatabase, nodeGPUMap map[string]string,
	k8sClient kubernetes.Interface, serviceName string) (string, error) {

	if cfg.PlacementStrategy == PlacementCapabilityOnly {
		return selectByCapability(cfg, gpuDB, nodeGPUMap, candidatesFromMap(nodeGPUMap))
	}

	// Stage 1: replica distribution
	// Counting only makes sense when the pod can be tied to a Knative service; if the label
	// is missing fall through to capability-only as a safe default.
	if serviceName == "" {
		log.Printf("[INFO/PLACEMENT] No Knative service label on pod, falling back to capability-only scoring")
		return selectByCapability(cfg, gpuDB, nodeGPUMap, candidatesFromMap(nodeGPUMap))
	}

	replicas, err := CountReplicasPerNode(k8sClient, cfg.TargetNamespace, serviceName, nodeGPUMap)
	if err != nil {
		// On API failure, degrade to capability-only instead of rejecting the pod.
		log.Printf("[ERROR] Replica count query failed (%v), falling back to capability-only scoring", err)
		return selectByCapability(cfg, gpuDB, nodeGPUMap, candidatesFromMap(nodeGPUMap))
	}

	minCount, candidates := minReplicaCandidates(replicas)
	log.Printf("[INFO/PLACEMENT] Service=%s replicas-per-node=%v min=%d candidates=%v",
		serviceName, replicas, minCount, candidates)

	// Stage 1 produced a winner: skip Prometheus entirely.
	if len(candidates) == 1 {
		log.Printf("[INFO/PLACEMENT] Selected node by load distribution: %s (replicas=%d)",
			candidates[0], minCount)
		return candidates[0], nil
	}

	// Stage 2: capability tiebreak on the surviving candidates.
	return selectByCapability(cfg, gpuDB, nodeGPUMap, candidates)
}

// selectByCapability runs the scoring on the given candidate nodes and returns the hostname with the
// highest final score.
func selectByCapability(cfg *Config, gpuDB *GPUDatabase, nodeGPUMap map[string]string,
	candidates []string) (string, error) {

	// Query GPU metrics for all nodes
	metrics, err := QueryGPUMetrics(cfg)
	if err != nil {
		return "", fmt.Errorf("failed to query GPU metrics: %w", err)
	}

	candidateSet := make(map[string]struct{}, len(candidates))
	for _, c := range candidates {
		candidateSet[c] = struct{}{}
	}

	var bestNode string
	var bestScore float64 = -1

	for hostname, metric := range metrics {
		// Skip nodes that were filtered out by Stage 1.
		if _, isCandidate := candidateSet[hostname]; !isCandidate {
			continue
		}

		dbModelName, exists := nodeGPUMap[hostname]
		if !exists {
			log.Printf("[ERROR] No GPU model mapping for node %s — skipping", hostname)
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

		// Geometric mean: expected throughput ≈ capability × availability.
		// capWeight controls how strongly the static (capability) score dominates.
		finalScore := math.Pow(dynamicScore, 1-capWeight) * math.Pow(staticScore, capWeight)

		log.Printf("[INFO/SCORING] %s: model=%s, dynamic=%.4f, static=%.4f, final=%.4f (capWeight=%.2f)",
			hostname, dbModelName, dynamicScore, staticScore, finalScore, capWeight)

		if finalScore > bestScore {
			bestScore = finalScore
			bestNode = hostname
		}
	}

	if bestNode == "" {
		return "", fmt.Errorf("no scoreable nodes found among candidates: all failed GPU model matching or scoring")
	}
	log.Printf("[INFO/SCORING] Selected node: %s (score=%.4f)", bestNode, bestScore)
	return bestNode, nil
}

// candidatesFromMap converts the node→GPU map into a flat hostname slice. Used when no Stage 1
// filtering is in effect and every known GPU node is a valid candidate.
func candidatesFromMap(nodeGPUMap map[string]string) []string {
	out := make([]string, 0, len(nodeGPUMap))
	for node := range nodeGPUMap {
		out = append(out, node)
	}
	return out
}

// minReplicaCandidates returns the lowest per-node replica count and the list of nodes tied at that
// minimum. The result is the input to Stage 2 (capability tiebreak): a single-element slice short-circuits
// the decision, a multi-element slice is passed to capability scoring.
func minReplicaCandidates(replicas map[string]int) (int, []string) {
	minCount := math.MaxInt
	for _, c := range replicas {
		if c < minCount {
			minCount = c
		}
	}

	candidates := make([]string, 0)
	for node, c := range replicas {
		if c == minCount {
			candidates = append(candidates, node)
		}
	}
	return minCount, candidates
}

// computeDynamicScore produces a [0.0, 1.0] score representing how "free" a node currently is, based on real-time
// DCGM metrics. The advertised total VRAM and the DCGM-reported total may differ by a few hundred MB due to
// driver-reserved memory; the clamp absorbs this discrepancy without distorting the score.
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

// matchGPUName tries to resolve a DCGM-reported GPU model name (e.g. "NVIDIA GeForce RTX 4060 Ti")
// to a database key (e.g. "GeForce RTX 4060 Ti 16 GB"). If an exact match is not found the algorithm
// picks the one whose advertised VRAM is closest to the DCGM-reported total VRAM.
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
