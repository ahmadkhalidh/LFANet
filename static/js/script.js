document.addEventListener("DOMContentLoaded", () => {
    const modal = document.getElementById("disclaimerModal");
    const checkbox = document.getElementById("consentCheck");
    const button = document.getElementById("acceptDisclaimer");

    checkbox.onchange = () => button.disabled = !checkbox.checked;
    button.onclick = () => {
        localStorage.setItem("accepted", "true");
        modal.style.display = "none";
    };

    if (localStorage.getItem("accepted") === "true") {
        modal.style.display = "none";
    }
});

function uploadImage() {
    if (localStorage.getItem("accepted") !== "true") {
        alert("Accept disclaimer first.");
        return;
    }

    const file = document.getElementById("imageInput").files[0];
    const explain = document.getElementById("toggleExplain").checked;

    const formData = new FormData();
    formData.append("image", file);
    formData.append("explain", explain);

    fetch("/predict", { method: "POST", body: formData })
    .then(r => r.json())
    .then(data => {
        document.getElementById("result").classList.remove("hidden");
        document.getElementById("predictedClass").innerText = data.prediction;
        document.getElementById("confidence").innerText =
            (data.confidence * 100).toFixed(2) + "%";

        if (data.explanation) {
            document.getElementById("camImage").src = data.cam_url;
            document.getElementById("camImage").classList.remove("hidden");
            document.getElementById("explanation").innerText = data.explanation;
            document.getElementById("pdfLink").href = data.pdf_url;
            document.getElementById("pdfLink").innerText = "Download PDF Report";
        }
    });
}
