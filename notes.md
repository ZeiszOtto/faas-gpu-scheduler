### Network addresses ###
rtx-master (Ubuntu 24.04 LTS):
    Wi-Fi:		192.168.1.10  (wlp0s20f3)
    Wired:		192.168.1.189 (eno1)
    Tailscale:	100.86.165.53 (tailscal0)

gtx-worker (Ubuntu 24.04 LTS):
    Wi-Fi:		-
    Wired:		192.168.1.86 (enp0s31f6)
    Tailscale:	-

Edge laptop (Windows 11 Pro 25H2):
    Wi-Fi:		192.168.1.30
    Wired:		192.168.1.82
    Tailscale:	100.93.205.76

Grafana port: 30952



### PowerShell RTT check for single inference ###
Measure-Command {
curl.exe -X POST -F "image=@2_person_0.53.jpg" `
http://face-detect.default.192-168-1-200.sslip.io/detect
}

1..10 | ForEach-Object {
$t = Measure-Command {
curl.exe -s -X POST -F "image=@2_person_0.53.jpg" `
      http://face-detect.default.192-168-1-200.sslip.io/detect | Out-Null
  }
  Write-Host "Request $_`: $($t.TotalMilliseconds) ms"
}


Measure-Command {
curl.exe -X POST -F "image=@1_vehicle_0.73.jpg" `
http://plate-detect.default.192-168-1-200.sslip.io/detect
}

1..10 | ForEach-Object {
$t = Measure-Command {
curl.exe -s -X POST -F "image=@120_vehicle_0.82.jpg" `
      http://plate-detect.default.192-168-1-200.sslip.io/detect | Out-Null
  }
  Write-Host "Request $_`: $($t.TotalMilliseconds) ms"
}



### iperf3 and ping baseline network test ###
LAN (~350/250 Mbps Wi-Fi connection):
    Transfer     Bandwidth
    140 MBytes   117 Mbits/sec 	sender
    140 MBytes   117 Mbits/sec 	receiver
    ---
    Transfer     Bandwidth      	
    112 MBytes  94.0 Mbits/sec 	sender
    112 MBytes  93.6 Mbits/sec 	receiver
    ---
    Approximate round trip times in milliseconds:
    Minimum = 2ms, Maximum = 6ms, Average = 4ms

VPN (~100 Mbps 5G mobile hotspot connection):
    Transfer     Bandwidth
    6.77 MBytes  5.68 Mbits/sec 	sender
    6.62 MBytes  5.56 Mbits/sec 	receiver
    ---
    Transfer     Bandwidth      	
    9.12 MBytes  7.65 Mbits/sec 	sender
    8.81 MBytes  7.39 Mbits/sec 	receiver
    ---
    Approximate round trip times in milli-seconds:
    Minimum = 19ms, Maximum = 64ms, Average = 42ms



### JPEG size network overhead ###
LAN (117 Mbps, 4 ms RTT):
    5 KB extra: 40 000 / 117 000 000 = 0.34 ms
    20 KB extra: 160 000 / 117 000 000 = 1.4 ms
    TCP slow start: ~2 RTT = 8 ms (fix overhead)

VPN (7 Mbps, 42 ms RTT):
    5 KB extra: 40 000 / 7 000 000 = 5.7 ms
    20 KB extra: 160 000 / 7 000 000 = 23 ms
    TCP slow start: ~2 RTT = 84 ms (fix overhead)



### Kubernetes node-forcing ###
kubectl patch ksvc face-detect -n default --type=merge -p '{
  "spec": {
    "template": {
      "spec": {
        "nodeSelector": {
          "kubernetes.io/hostname": "<node>"
        }
      }
    }
  }
}'


### Pod warmup commands ###

$elapsed = Measure-Command {
    $response = curl.exe -s -X POST --data-binary "@C:\Users\fragm\GolandProjects\faas-gpu-scheduler\yolo-output\1_person_0.45.jpg" `
        -H "Content-Type: image/jpeg" `
        http://face-detect.default.192-168-1-200.sslip.io/detect
    Write-Host "Response: $response"
}
Write-Host "Warmup latency: $($elapsed.TotalMilliseconds) ms" -ForegroundColor Yellow
Start-Sleep -Seconds 8
Write-Host "Ready for measurement." -ForegroundColor Green


$elapsed = Measure-Command {
    $response = curl.exe -s -X POST --data-binary "@C:\Users\fragm\GolandProjects\faas-gpu-scheduler\yolo-output\112_vehicle_0.85.jpg" `
        -H "Content-Type: image/jpeg" `
        http://plate-detect.default.192-168-1-200.sslip.io/detect
    Write-Host "Response: $response"
}
Write-Host "Warmup latency: $($elapsed.TotalMilliseconds) ms" -ForegroundColor Yellow
Start-Sleep -Seconds 8
Write-Host "Ready for measurement." -ForegroundColor Green

### PowerShell scripting ###

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass