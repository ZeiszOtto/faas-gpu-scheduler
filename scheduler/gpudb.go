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

func (db *GPUDatabase) ValidatePreset(presetName string) error {
	if _, exists := db.ScoringPresets[presetName]; !exists {
		available := make([]string, 0, len(db.ScoringPresets))
		for name := range db.ScoringPresets {
			available = append(available, name)
		}
		return fmt.Errorf("scoring preset %q not found in GPU database", presetName)
	}
	return nil
}

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

func (db *GPUDatabase) ListAvailableGPUs() []string {
	names := make([]string, 0, len(db.GPUs))
	for name := range db.GPUs {
		names = append(names, name)
	}
	return names
}

func (db *GPUDatabase) LogGPUInfo(modelName string, presetName string, tensorEnabled bool) {
	gpu, exists := db.GPUs[modelName]
	if !exists {
		log.Printf("[ERROR] GPU model %q not found in database", modelName)
		return
	}

	score, err := db.GetGPUScore(modelName, presetName, tensorEnabled)
	if err != nil {
		log.Printf("[ERROR] Could not compute score for %q: %v", modelName, err)
		return
	}

	log.Printf("[INFO/GPUDB] %s: arch=%s, VRAM=%.0fGB, BW=%.0f GB/s, TC=%d, FP32=%.1f TFLOPS → score=%.4f (preset=%s, tensor=%t)",
		modelName, gpu.Architecture, gpu.MemorySizeGB, gpu.MemoryBandwidthGBs,
		gpu.TensorCores, gpu.FP32TFLOPS, score, presetName, tensorEnabled)
}
