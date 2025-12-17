const form = document.getElementById("predictionForm");
const resultsCard = document.getElementById("resultsCard");
const errorAlert = document.getElementById("errorAlert");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const title = titleInput().value.trim();
  const body = bodyInput().value.trim();

  hideError();
  hideResults();

  if (!title && !body) {
    return showError("Please provide a title or article body.");
  }

  setLoading(true);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, body })
    });

    const data = await res.json();
    if (!res.ok) throw new Error(data.error);

    renderResults(data);

  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
});

function renderResults({ label, probability }) {
  // probability = probability of FAKE (number between 0 and 1)
  const fakeProb = Math.max(0, Math.min(1, probability));
  const fakePercentNum = Math.round(fakeProb * 100);
  const ring = document.querySelector(".confidence-ring");

  const isFake = label === "FAKE";

  // ---- Headline (trust backend decision) ----
  document.getElementById("resultLabel").textContent =
    isFake ? "Likely Fake News" : "Likely Real News";

  // ---- Human-safe percentage text ----
  const fakePercentText =
    fakeProb < 0.01 ? "<1%" :
    fakeProb > 0.99 ? ">99%" :
    `${fakePercentNum}%`;

  document.getElementById("confidenceValue").textContent = fakePercentText;

  // ---- Logically correct explanation ----
  document.getElementById("resultSummary").textContent =
    isFake
      ? `This article has a ${fakePercentText} probability of being fake and is therefore likely unreliable.`
      : `This article has a ${fakePercentText} probability of being fake and is therefore likely real.`;

  // ---- Ring visualization (numeric only) ----
  ring.style.setProperty("--percent", `${fakePercentNum}%`);
  ring.style.background = `conic-gradient(
    ${isFake ? "#dc3545" : "#28a745"} ${fakePercentNum}%,
    rgba(255,255,255,0.15) 0
  )`;

  resultsCard.classList.remove("d-none");
  resultsCard.scrollIntoView({ behavior: "smooth" });
}

function setLoading(state) {
  const btn = document.getElementById("submitBtn");
  btn.disabled = state;
  btn.innerHTML = state
    ? `<span class="spinner-border spinner-border-sm"></span> Analyzing…`
    : `<i class="bi bi-cpu"></i> Analyze Credibility`;
}

function showError(msg) {
  document.getElementById("errorMessage").textContent = msg;
  errorAlert.classList.remove("d-none");
}

function hideError() {
  errorAlert.classList.add("d-none");
}

function hideResults() {
  resultsCard.classList.add("d-none");
}

const titleInput = () => document.getElementById("title");
const bodyInput = () => document.getElementById("body");