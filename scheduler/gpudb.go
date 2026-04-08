package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

type GPUDatabase struct {
	Metadata       dbMetadata               `json:"metadata"`
	ScoringPresets map[string]ScoringPreset `json:"scoring_presets"`
	GPUs           map[string]GPUSpec       `json:"gpus"`
}

type dbMetadata struct {
	Timestamp string `json:"timestamp"`
	Source    string `json:"source"`
	GPUCount  int    `json:"gpu_count"`
}

type ScoringPreset struct {
	Description      string  `json:"description"`
	BandwidthWeight  float64 `json:"bandwidth_weight"`
	VRAMWeight       float64 `json:"vram_weight"`
	TensorWeight     float64 `json:"tensor_weight"`
	FP32Weight       float64 `json:"fp32_weight"`
	MemoryTypeWeight float64 `json:"memory_type_weight"`
}

type GPUSpec struct {
	Architecture       string  `json:"architecture"`
	MemorySizeGB       float64 `json:"memory_size_gb"`
	MemoryType         string  `json:"memory_type"`
	MemoryTypeOrdinal  int     `json:"memory_type_ordinal"`
	MemoryBandwidthGBs float64 `json:"memory_bandwidth_gb_s"`
	TensorCores        int     `json:"tensor_cores"`
	CUDACores          int     `json:"cuda_cores"`
	FP32TFLOPS         float64 `json:"fp32_tflops"`

	NormBandwidth  float64 `json:"norm_bandwidth"`
	NormVRAM       float64 `json:"norm_vram"`
	NormTensor     float64 `json:"norm_tensor"`
	NormFP32       float64 `json:"norm_fp32"`
	NormMemoryType float64 `json:"norm_memory_type"`
}

// LoadGPUDatabase reads and parses the GPU capability database from the given JSON file path,
// returning a fully populated GPUDatabase pointer.
func LoadGPUDatabase(path string) (*GPUDatabase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("error reading GPU database file %s: %v", path, err)
	}

	var db GPUDatabase
	if err := json.Unmarshal(data, &db); err != nil {
		return nil, fmt.Errorf("error parsing GPU database file %s: %v", path, err)
	}

	if len(db.GPUs) == 0 {
		return nil, fmt.Errorf("no GPU entries found in the database")
	}
	if len(db.ScoringPresets) == 0 {
		return nil, fmt.Errorf("no scoring presets found in the database")
	}

	log.Printf("[INFO/GPUDB] GPU database successfully loaded with %d GPU entries.", len(db.GPUs))

	return &db, nil
}

// ValidatePreset checks whether the given preset name exists in the loaded scoring presets map.
// Returns nil on success, or an error listing the available preset names.
func (db *GPUDatabase) ValidatePreset(presetName string) error {
	if _, exists := db.ScoringPresets[presetName]; !exists {
		available := make([]string, 0, len(db.ScoringPresets))
		for name := range db.ScoringPresets {
			available = append(available, name)
		}
		return fmt.Errorf("scoring preset %q not found in GPU database (available: %v)", presetName, available)
	}
	return nil
}

// GetGPUScore computes the static capability score for a single GPU model under the given scoring preset.
// If tensorEnabled is false, the tensor core weight is removed from the formula and the remaining four
// weights are rescaled proportionally so they still sum to 1.0
func (db *GPUDatabase) GetGPUScore(modelName string, presetName string, tensorEnabled bool) (float64, error) {
	// Look up the GPU model in the database
	gpu, exists := db.GPUs[modelName]
	if !exists {
		return 0, fmt.Errorf("GPU model %q not found in database", modelName)
	}

	// Look up the scoring preset
	preset, exists := db.ScoringPresets[presetName]
	if !exists {
		return 0, fmt.Errorf("scoring preset %q not found in database", presetName)
	}

	// Creating a copy of the preset weights
	bwWeight := preset.BandwidthWeight
	vramWeight := preset.VRAMWeight
	tensorWeight := preset.TensorWeight
	fp32Weight := preset.FP32Weight
	memTypeWeight := preset.MemoryTypeWeight

	// If tensor scoring is disabled, redistribute the tensor weight proportionally
	if !tensorEnabled {
		activeSum := bwWeight + vramWeight + fp32Weight + memTypeWeight
		if activeSum > 0 {
			scale := 1.0 / activeSum
			bwWeight *= scale
			vramWeight *= scale
			fp32Weight *= scale
			memTypeWeight *= scale
		}
		tensorWeight = 0
	}

	// Compute the weighted sum of normalized metric values
	score := bwWeight*gpu.NormBandwidth +
		vramWeight*gpu.NormVRAM +
		tensorWeight*gpu.NormTensor +
		fp32Weight*gpu.NormFP32 +
		memTypeWeight*gpu.NormMemoryType

	return score, nil
}
