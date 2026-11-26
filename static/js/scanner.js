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
    if (!window.isSecureContext) {
      showError("Camera access requires http://127.0.0.1:5000, http://localhost:5000, or HTTPS. Switch to http://127.0.0.1:5000/scanner or upload an image.");
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      showError("Camera access is not available in this browser. You can upload an image instead.");
      return;
    }
    stopCamera();
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false,
        });
      } catch (constraintErr) {
        if (constraintErr.name === "OverconstrainedError" || constraintErr.name === "NotFoundError") {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } else {
          throw constraintErr;
        }
      }
      video.srcObject = stream;
      await video.play().catch(() => {});
      cameraPanel.hidden = false;
    } catch (cameraError) {
      if (cameraError.name === "NotAllowedError" || cameraError.name === "PermissionDeniedError") {
        showError("Camera permission was denied. Click the camera/tune icon in your browser's address bar to Allow camera access (and check Windows Privacy -> Camera), or upload an image.");
      } else if (cameraError.name === "NotFoundError" || cameraError.name === "DevicesNotFoundError") {
        showError("No camera was detected on this device. Please upload an image instead.");
      } else if (cameraError.name === "NotReadableError" || cameraError.name === "TrackStartError") {
        showError("Camera is currently in use by another app. Close the other app and try again, or upload an image.");
      } else {
        showError("Could not open the camera. You can upload an image instead.");
      }
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

  let lastScanId = null;

  function renderResult(data) {
    lastScanId = data.id || null;
    const result = root.querySelector("[data-result]");
    const uncertain = data.uncertain;
    const level = (data.confidence_level || (uncertain ? "low" : "high")).toUpperCase();
    root.querySelector("[data-result-status]").textContent = uncertain
      ? "LOW CONFIDENCE"
      : `${level} CONFIDENCE`;
    root.querySelector("[data-result-status]").className = `result-status${uncertain ? " status-uncertain" : ""}`;
    root.querySelector("[data-result-category]").textContent = uncertain
      ? "Not sure yet"
      : data.recommendation.label;
    const possible = root.querySelector("[data-result-possible]");
    if (uncertain && Array.isArray(data.top_3) && data.top_3.length > 0) {
      const topText = data.top_3.map((t) => `${t.category} (${Math.round(t.confidence * 100)}%)`).join(", ");
      possible.textContent = `${data.message || "Try a closer photo."} Top guesses: ${topText}`;
      possible.hidden = false;
    } else if (uncertain) {
      possible.textContent = `Possible match: ${data.recommendation.label}`;
      possible.hidden = false;
    } else {
      possible.hidden = true;
      possible.textContent = "";
    }
    root.querySelector("[data-confidence]").textContent = `${Math.round(data.confidence * 100)}%`;
    root.querySelector("[data-confidence-bar]").style.width = `${Math.round(data.confidence * 100)}%`;
    root.querySelector("[data-disposal]").textContent = data.recommendation.disposal;
    root.querySelector("[data-recycling]").textContent =
      `${data.recommendation.recycling} ${data.recommendation.reuse || ""}`.trim();
    root.querySelector("[data-result-image]").src = data.image_url;

    const fbStatus = root.querySelector("[data-feedback-status]");
    const fbSelect = root.querySelector("[data-feedback-select]");
    const fbSubmit = root.querySelector("[data-feedback-submit]");
    const fbConsent = root.querySelector("[data-feedback-consent]");
    if (fbStatus) fbStatus.hidden = true;
    if (fbSelect) fbSelect.hidden = true;
    if (fbSubmit) fbSubmit.hidden = true;
    if (fbConsent) fbConsent.checked = false;

    result.hidden = false;
    result.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function sendFeedback(isCorrect, correctedCategory) {
    const fbStatus = root.querySelector("[data-feedback-status]");
    const fbConsent = root.querySelector("[data-feedback-consent]");
    if (!fbConsent?.checked) {
      fbStatus.textContent = "Please check the opt-in consent box first.";
      fbStatus.hidden = false;
      return;
    }
    if (!lastScanId) return;
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scan_id: lastScanId,
          is_correct: isCorrect,
          corrected_category: correctedCategory,
          consent: true,
        }),
      });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error || "Could not save feedback.");
      fbStatus.textContent = `Thanks! Saved under '${out.label}' for future training.`;
      fbStatus.hidden = false;
    } catch (err) {
      fbStatus.textContent = err.message;
      fbStatus.hidden = false;
    }
  }

  root.querySelector("[data-feedback-yes]")?.addEventListener("click", () => sendFeedback(true, null));
  root.querySelector("[data-feedback-no]")?.addEventListener("click", async () => {
    const fbSelect = root.querySelector("[data-feedback-select]");
    const fbSubmit = root.querySelector("[data-feedback-submit]");
    if (!fbSelect.options.length) {
      try {
        const res = await fetch("/api/categories");
        const data = await res.json();
        (data.categories || ["cardboard", "glass", "metal", "paper", "plastic", "trash"]).forEach((c) => {
          const opt = document.createElement("option");
          opt.value = c;
          opt.textContent = c;
          fbSelect.appendChild(opt);
        });
      } catch (_) {}
    }
    fbSelect.hidden = false;
    fbSubmit.hidden = false;
  });
  root.querySelector("[data-feedback-submit]")?.addEventListener("click", () => {
    const fbSelect = root.querySelector("[data-feedback-select]");
    sendFeedback(false, fbSelect.value);
  });
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
