# faas-gpu-scheduler

GPU-aware Kubernetes scheduler for FaaS / latency-sensitive AI workloads,
implemented as a **Mutating Admission Webhook** in Go and deployed alongside
Knative Serving on a heterogeneous GPU cluster.

The webhook complements Knative's Pod Autoscaler (KPA): KPA decides *how many*
replicas a service needs, the webhook decides *which GPU node* each replica
lands on. Placement is driven by real-time NVIDIA DCGM metrics from Prometheus
plus a static GPU capability score derived from a built-in GPU database.

## Architecture

```
┌────────────────────┐     person crops      ┌──────────────────────────┐
│  client_processor  │ ───────────────────▶ │  face-detect (Knative)   │
│ (YOLOv8 + tracker) │     vehicle crops    │  plate-detect (Knative)  │
└────────────────────┘ ───────────────────▶ └──────────────────────────┘
        │                                              ▲
        │                                              │ Pod CREATE
        ▼                                              │
┌────────────────────┐    AdmissionReview    ┌─────────┴────────────────┐
│   kube-apiserver   │ ────────────────────▶ │  gpu-scheduler-webhook   │
└────────────────────┘ ◀──────────────────── │  (Mutating Webhook, Go)  │
                          nodeAffinity patch └──────────┬───────────────┘
                                                        │ PromQL
                                                        ▼
                                              ┌───────────────────┐
                                              │   Prometheus +    │
                                              │   DCGM Exporter   │
                                              └───────────────────┘
```

---

## Prerequisites

- Ubuntu 22.04+ on every node, with NVIDIA drivers installed
- Kubernetes, Knative Serving + an ingress (e.g. Kourier)
- MetalLB (or another LoadBalancer provider)
- NVIDIA GPU Operator with `driver.enabled=false`
- kube-prometheus-stack + DCGM Exporter
- `kubectl`, `helm`, `openssl`, and a container build tool

---

## 1. Cluster setup

Run the node bootstrap script on **every** node (control-plane and workers):

```bash
sudo bash scripts/containerd_setup.sh
```

This installs and configures containerd with `SystemdCgroup=true`, loads the
required kernel modules (`overlay`, `br_netfilter`), applies the necessary
sysctl settings, and disables swap.

After bootstrap, initialize the cluster with `kubeadm init` on the
control-plane node and join the workers. Install a CNI (e.g. Flannel), then
deploy Knative Serving, the ingress, MetalLB, the NVIDIA GPU Operator,
Prometheus, and DCGM Exporter via their official Helm charts or manifests.
Knative's domain config should be set to a hostname that resolves to your
LoadBalancer VIP (e.g. via `sslip.io`).

---

## 2. Deploy the scheduler

### 2.1 (Optional) Regenerate the GPU database

`gpu_database.json` is baked into the scheduler image and only needs to be
regenerated when GPU coverage changes.

```bash
cd scheduler/
pip install "dbgpu[fuzz]"
python generate_gpudb.py -o gpu_database.json
```

### 2.2 Build and push the scheduler image

```bash
cd scheduler/
docker build -t <your-registry>/scheduler:latest .
docker push <your-registry>/scheduler:latest
```

Update the `image:` field in `scheduler-deployment.yaml` accordingly.

### 2.3 Generate TLS certificates and the CA bundle

The webhook must be served over TLS. The helper script creates a self-signed
CA, signs the server certificate for the webhook Service DNS name, stores
both in a `kubernetes.io/tls` Secret, and patches the `caBundle:` field in
`scheduler-config.yaml`.

```bash
cd scheduler/deploy/
WEBHOOK_CONFIG=$(pwd)/scheduler-config.yaml \
    bash ../../scripts/generate_certs.sh
```

### 2.4 Apply the manifests (order matters)

```bash
kubectl apply -f scheduler-rbac.yaml         # ServiceAccount, ClusterRole, Binding
kubectl apply -f scheduler-configmap.yaml    # Scheduler runtime parameters
kubectl apply -f scheduler-service.yaml      # ClusterIP service
kubectl apply -f scheduler-deployment.yaml   # Webhook pod
kubectl apply -f scheduler-config.yaml       # MutatingWebhookConfiguration
```

Verify:

```bash
kubectl -n gpu-scheduler get pods
kubectl -n gpu-scheduler logs deploy/gpu-scheduler-webhook
```

### 2.5 Key ConfigMap parameters

| Key                   | Meaning                                                   |
|-----------------------|-----------------------------------------------------------|
| `PROMETHEUS_URL`      | Prometheus endpoint exposing DCGM metrics                 |
| `METRIC_WINDOW`       | `avg_over_time` smoothing window for PromQL               |
| `SCORE_PRESET`        | `inference` / `balanced` / `training`                     |
| `CAPABILITY_WEIGHT`   | Weight of static capability vs. dynamic availability [0,1]|
| `PLACEMENT_STRATEGY`  | `capability_only` or `load_balanced`                      |
| `SCHEDULING_ENABLED`  | `false` = compute and log only, do not patch              |

---

## 3. Deploy the inference services

Both services are FastAPI apps exposing `POST /detect` that accept a raw JPEG
body and return a JSON result. They are deployed as Knative Services with
`nvidia.com/gpu: "1"` resource limits.

```bash
cd services/face-detect/
docker build -t <your-registry>/face-detect:latest .
docker push <your-registry>/face-detect:latest
kubectl apply -f face-detect-ksvc.yaml

cd ../plate-detect/
docker build -t <your-registry>/plate-detect:latest .
docker push <your-registry>/plate-detect:latest
kubectl apply -f plate-detect-ksvc.yaml
```

Update the `image:` field in each `*-ksvc.yaml` to match your registry.

Verify the endpoints:

```bash
kubectl get ksvc -n default
curl -X POST --data-binary @sample.jpg \
    -H "Content-Type: image/jpeg" \
    http://face-detect.default.<your-domain>/detect
```

---

## 4. Edge pipeline — `client_processor.py`

Runs on the **edge node** (typically a laptop), not inside Kubernetes. Reads
a video file, runs YOLOv8 with tracking on every frame, and dispatches
person/vehicle crops to the appropriate Knative service over HTTP.

### 4.1 Install

```bash
cd client/
pip install ultralytics opencv-python requests pyyaml
```

### 4.2 Configure (`config.yaml`)

Key fields:

```yaml
mode: "save"   # "save" -> write crops to disk, "http" -> POST to services
services:
  face_detect:  "http://face-detect.default.<your-domain>/detect"
  plate_detect: "http://plate-detect.default.<your-domain>/detect"
video:
  source: "../yolo-input/input_dataset.mp4"
  frame_skip: 1
filtering:
  min_person_height: 100   # skip persons shorter than this (px)
  min_vehicle_width: 200   # skip vehicles narrower than this (px)
crop_processing:
  enabled: true
  letterbox: true                  # aspect-aware resize + padding
  person:  { target_size: [240, 440] }
  vehicle: { target_size: [400, 400] }
tracking:
  cooldown_frames: 15      # per-track-ID dispatch throttle
yolo:
  model: "yolov8n.pt"
  confidence_threshold: 0.4
  device: "cuda:0"
output:
  save_dir: "../yolo-output"
```

### 4.3 Run

```bash
# Save crops to disk for later replay by load_simulator.py
python client_processor.py config.yaml          # mode: "save"

# Live dispatch to the cluster
# (set mode: "http" in config.yaml first)
python client_processor.py config.yaml
```

`save` mode produces filenames of the form `<frame>_<class>_<conf>.jpg` plus
a `results.csv` log; these crops are the input to the load simulator.

---

## 5. Load simulator — `load_simulator.py`

Replays the pre-generated crops at a controlled rate to drive reproducible
benchmark runs. Uses a thread pool for concurrency and a shared
`requests.Session` for HTTP keep-alive.

### 5.1 Basic usage

```bash
# 300 requests at 5 req/s, all classes
python load_simulator.py --total 300 --rate 5

# Burst mode (no pacing, fire as fast as the pool allows)
python load_simulator.py --total 100 --rate 0

# Higher concurrency
python load_simulator.py --total 1000 --rate 10 --workers 100

# Hit only one service
python load_simulator.py --total 500 --rate 5 --class person   # face-detect
python load_simulator.py --total 500 --rate 5 --class vehicle  # plate-detect

# Deterministic crop interleaving across runs
python load_simulator.py --total 500 --rate 5 --shuffle-seed 42
```

### 5.2 Output

Each run produces a timestamped CSV in `output.save_dir`:

```
yolo-output/load_results_<timestamp>.csv
```

Columns: `submitted_at, completed_at, frame, class, confidence, target,
status_code, latency_ms, response, error`.

### 5.3 Warmup before benchmarks

The first request after idle hits the Knative activator's cold path, which
is significantly slower than the warm path. Always send a few warmup
requests to both services before each measurement run:

```bash
python load_simulator.py --total 20 --rate 2   # discard this CSV
python load_simulator.py --total 500 --rate 5  # measurement run
```