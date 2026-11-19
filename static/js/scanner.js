(() => {
  const root = document.querySelector("[data-scanner]");
  if (!root) return;
  const fileInput = root.querySelector("[data-file-input]");
  const dropZone = root.querySelector("[data-drop-zone]");
  const emptyState = root.querySelector("[data-empty-state]");
  const previewWrap = root.querySelector("[data-preview-wrap]");
  const preview = root.querySelector("[data-preview]");
  const analyze = root.querySelector("[data-analyze]");
  const error = root.querySelector("[data-error]");
  const cameraPanel = root.querySelector("[data-camera-panel]");
  const video = root.querySelector("[data-video]");
  let selectedFile = null;
  let stream = null;
  let previewUrl = null;

  const showError = (message) => {
    error.textContent = message;
    error.hidden = false;
  };
  const stopCamera = () => {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    cameraPanel.hidden = true;
  };
  const setFile = (file) => {
    error.hidden = true;
    if (!file) return;
    if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
      showError("Choose a JPEG, PNG, or WebP image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      showError("This image is larger than 10 MB.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    root.querySelector("[data-preview-name]").textContent = file.name || "Camera capture";
    emptyState.hidden = true;
    previewWrap.hidden = false;
    analyze.disabled = false;
    root.querySelector("[data-result]").hidden = true;
  };
  root.querySelector("[data-browse]").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  root.querySelector("[data-remove]").addEventListener("click", () => {
    selectedFile = null;
    fileInput.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.src = "";
    previewWrap.hidden = true;
    emptyState.hidden = false;
    analyze.disabled = true;
  });
  ["dragenter", "dragover"].forEach((type) => dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.add("drop-active");
  }));
  ["dragleave", "drop"].forEach((type) => dropZone.addEventListener(type, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drop-active");
  }));
  dropZone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

  root.querySelector("[data-camera]").addEventListener("click", async () => {
    error.hidden = true;
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Camera access is not available in this browser. You can upload an image instead.");
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
      video.srcObject = stream;
      cameraPanel.hidden = false;
    } catch (cameraError) {
      showError(cameraError.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access or upload an image."
        : "Could not open the camera. You can upload an image instead.");
    }
  });
  root.querySelector("[data-camera-close]").addEventListener("click", stopCamera);
  root.querySelector("[data-capture]").addEventListener("click", () => {
    if (!video.videoWidth || !video.videoHeight) {
      showError("The camera is still starting. Try capturing again in a moment.");
      return;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (!blob) {
        showError("The camera image could not be captured. Please try again.");
        return;
      }
      setFile(new File([blob], "camera-capture.jpg", { type: "image/jpeg" }));
      stopCamera();
    }, "image/jpeg", 0.92);
  });

  root.querySelector("[data-analyze]").addEventListener("click", async () => {
    if (!selectedFile) return;
    const loading = root.querySelector("[data-loading]");
    const button = root.querySelector("[data-analyze]");
    const formData = new FormData();
    formData.append("image", selectedFile);
    error.hidden = true;
    loading.hidden = false;
    button.disabled = true;
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "The image could not be analyzed.");
      renderResult(data);
    } catch (requestError) {
      showError(requestError.message || "The server could not analyze this image.");
    } finally {
      loading.hidden = true;
      button.disabled = !selectedFile;
    }
  });

  function renderResult(data) {
    const result = root.querySelector("[data-result]");
    const uncertain = data.uncertain;
    root.querySelector("[data-result-status]").textContent = uncertain ? "LOW CONFIDENCE" : "MATCH FOUND";
    root.querySelector("[data-result-status]").className = `result-status${uncertain ? " status-uncertain" : ""}`;
    root.querySelector("[data-result-category]").textContent = uncertain
      ? "Not sure yet"
      : data.recommendation.label;
    const possible = root.querySelector("[data-result-possible]");
    possible.hidden = !uncertain;
    possible.textContent = uncertain ? `Possible match: ${data.recommendation.label}` : "";
    root.querySelector("[data-confidence]").textContent = `${Math.round(data.confidence * 100)}%`;
    root.querySelector("[data-confidence-bar]").style.width = `${Math.round(data.confidence * 100)}%`;
    root.querySelector("[data-disposal]").textContent = data.recommendation.disposal;
    root.querySelector("[data-recycling]").textContent =
      `${data.recommendation.recycling} ${data.recommendation.reuse || ""}`.trim();
    root.querySelector("[data-result-image]").src = data.image_url;
    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  root.querySelector("[data-scan-again]").addEventListener("click", () => {
    root.querySelector("[data-result]").hidden = true;
    root.querySelector("[data-remove]").click();
    root.querySelector(".scanner-card").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  fetch("/api/status")
    .then((response) => response.json())
    .then((status) => {
      root.querySelector("[data-model-status]").textContent = status.model_available
        ? "Model ready to scan"
        : "Model not installed";
      root.querySelector("[data-model-message]").textContent = status.message;
    })
    .catch(() => {
      root.querySelector("[data-model-status]").textContent = "Status unavailable";
      root.querySelector("[data-model-message]").textContent = "The model status could not be checked. Please try refreshing the page.";
    });
})();
