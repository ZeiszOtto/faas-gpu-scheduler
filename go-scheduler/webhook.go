package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"

	admissionv1 "k8s.io/api/admission/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
)

type PatchOperation struct {
	Op    string      `json:"op"`
	Path  string      `json:"path"`
	Value interface{} `json:"value,omitempty"`
}

// handleMutate ...
func handleMutate(cfg *Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// Read incoming HTTP Request
		body, err := io.ReadAll(r.Body)
		if err != nil {
			log.Printf("[ERROR] An error occurred during reading request body: %v", err)
			http.Error(w, "Error reading body", http.StatusBadRequest)
			return
		}
		defer r.Body.Close()

		// JSON Deserialization into AdmissionReview
		var admissionReview admissionv1.AdmissionReview
		if err := json.Unmarshal(body, &admissionReview); err != nil {
			log.Printf("[ERROR] An error occurred during parsing AdmissionReview: %v", err)
			http.Error(w, "Error parsing JSON", http.StatusBadRequest)
			return
		}

		// Request sanity checking
		req := admissionReview.Request
		if req == nil {
			log.Printf("[ERROR] Empty AdmissionReview request")
			http.Error(w, "Empty AdmissionReview request", http.StatusBadRequest)
			return
		}

		log.Printf("[INFO] Webhook call: Operation=%s Namespace=%s Name=%s",
			req.Operation, req.Namespace, req.Name)

		// Pod extraction
		var pod corev1.Pod
		if err := json.Unmarshal(req.Object.Raw, &pod); err != nil {
			log.Printf("[ERROR] An error occurred during parsing Pod object: %s", err)
			sendResponse(w, req.UID, false, "Error parsing pod", nil)
			return
		}

		// Pod relevance checking
		if req.Namespace != cfg.TargetNamespace {
			log.Printf("[INFO] Scheduling disregarded on pod: Not part of %s namespace", cfg.TargetNamespace)
			sendResponse(w, req.UID, true, "Not part of target namespace", nil)
			return
		}
		if !requestsGPU(&pod) {
			log.Printf("[INFO] Scheduling disregarded on pod: GPU resources not requested")
			sendResponse(w, req.UID, true, "GPU resources not requested", nil)
			return
		}

		// Node selection
		selectedNode, err := selectNode(cfg)
		if err != nil {
			log.Printf("[ERROR] Unsuccessful node selection: %s — pod allowed without patching", err)
			sendResponse(w, req.UID, true, "Node selection error [fallback]", nil)
			return
		}
		log.Printf("[INFO] Selected node: %s | Pod: %s/%s", selectedNode, req.Namespace, pod.GenerateName)

		// Building and dispatching JSON Patch
		patch := buildNodeAffinityPatch(selectedNode)
		patchBytes, err := json.Marshal(patch)
		if err != nil {
			log.Printf("[ERROR] An error occurred during marshalling patch: %v", err)
			sendResponse(w, req.UID, false, "An error occurred during marshalling patch", nil)
			return
		}
		log.Printf("[INFO] Patch completed for %s pod with affinity for %s node", pod.GenerateName, selectedNode)
		sendResponse(w, req.UID, true, "Patch completed with affinity for"+selectedNode, patchBytes)
	}
}

// requestsGPU checks if any container within the pod requests GPU resources and returns true if so.
func requestsGPU(pod *corev1.Pod) bool {
	for _, container := range pod.Spec.Containers {
		if container.Resources.Limits != nil {
			if _, exists := container.Resources.Limits["nvidia.com/gpu"]; exists {
				return true
			}
		}
	}
	return false
}

// buildNodeAffinityPatch ...
func buildNodeAffinityPatch(nodeName string) []PatchOperation {
	affinity := corev1.Affinity{
		NodeAffinity: &corev1.NodeAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []corev1.PreferredSchedulingTerm{
				{
					// Explicitly recommends node
					Weight: 100,
					Preference: corev1.NodeSelectorTerm{
						MatchExpressions: []corev1.NodeSelectorRequirement{
							{
								Key:      "kubernetes.io/hostname",
								Operator: corev1.NodeSelectorOpIn,
								Values:   []string{nodeName},
							},
						},
					},
				},
			},
		},
	}

	return []PatchOperation{
		{
			Op:    "add",
			Path:  "/spec/affinity",
			Value: affinity,
		},
	}
}

// sendResponse Sends the AdmissionReview response to the API server
func sendResponse(w http.ResponseWriter, uid interface{}, allowed bool, message string, patch []byte) {
	uidStr := fmt.Sprintf("%v", uid)

	response := admissionv1.AdmissionReview{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "admission.k8s.io/v1",
			Kind:       "AdmissionReview",
		},
		Response: &admissionv1.AdmissionResponse{
			UID:     types.UID(uidStr),
			Allowed: allowed,
			Result: &metav1.Status{
				Message: message,
			},
		},
	}

	if patch != nil {
		response.Response.PatchType = new(admissionv1.PatchTypeJSONPatch)
		response.Response.Patch = patch
	}

	responseBytes, err := json.Marshal(response)
	if err != nil {
		log.Printf("[ERROR] An error occurred during marshalling response: %s", err)
		http.Error(w, "Error marshalling response", http.StatusInternalServerError)
	}

	w.Header().Set("Content-Type", "application/json")
	if _, err := w.Write(responseBytes); err != nil {
		log.Printf("[ERROR] Failed to write response: %s", err)
	}
}

// selectNode ...
func selectNode(cfg *Config) (string, error) {
	log.Printf("[STUB] SelectNode called — temporary: rtx-master")
	return "rtx-master", nil
}
