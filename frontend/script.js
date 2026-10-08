document.addEventListener("DOMContentLoaded", () => {
    const statusDot = document.getElementById("statusDot");
    const backendStatus = document.getElementById("backendStatus");
    const phaseDetail = document.getElementById("phaseDetail");
    const endpointUrl = document.getElementById("endpointUrl");

    // Determine API URL based on how the page is being served
    const isHttp = window.location.protocol === "http:" || window.location.protocol === "https:";
    const baseUrl = isHttp ? window.location.origin : "http://127.0.0.1:5000";
    const healthUrl = `${baseUrl}/api/health`;

    if (endpointUrl) {
        endpointUrl.textContent = healthUrl;
    }

    async function checkHealth() {
        try {
            const response = await fetch(healthUrl, {
                method: "GET",
                headers: {
                    "Accept": "application/json"
                }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.status === "ok") {
                statusDot.className = "status-indicator connected";
                backendStatus.innerHTML = 'Backend Status: <span class="connected-text">Connected</span>';
                if (phaseDetail) {
                    phaseDetail.textContent = data.phase || "Phase 1 - Network Traffic Intelligence";
                }
            } else {
                throw new Error("Unexpected response format");
            }
        } catch (error) {
            console.error("Health check failed:", error);
            statusDot.className = "status-indicator error";
            backendStatus.innerHTML = 'Backend Status: <span class="error-text">Offline / Error</span>';
            if (phaseDetail) {
                phaseDetail.textContent = "Could not reach Flask server";
            }
        }
    }

    // Run health check on initial load
    checkHealth();
});
