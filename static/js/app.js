// notice2action-ai | TrustExtract-N Frontend Application Logic

let sampleNotices = [];
let currentAnalysisResult = null;

document.addEventListener("DOMContentLoaded", () => {
  fetchSampleNotices();

  const form = document.getElementById("analyze-form");
  if (form) {
    form.addEventListener("submit", handleFormSubmit);
  }
});

// Fetch pre-loaded sample notices from API
async function fetchSampleNotices() {
  try {
    const res = await fetch("/api/samples");
    if (res.ok) {
      sampleNotices = await res.json();
      renderSampleChips();
      // Load first sample by default in textarea
      if (sampleNotices.length > 0) {
        loadSampleNotice(sampleNotices[0].id);
      }
    }
  } catch (err) {
    console.error("Failed to fetch sample notices:", err);
  }
}

// Render sample notice chips
function renderSampleChips() {
  const container = document.getElementById("samples-bar");
  if (!container) return;

  container.innerHTML = "";
  sampleNotices.forEach((sample, idx) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `sample-chip ${idx === 0 ? "active" : ""}`;
    chip.innerHTML = `<span>${sample.title}</span>`;
    chip.onclick = () => {
      document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      loadSampleNotice(sample.id);
    };
    container.appendChild(chip);
  });
}

// Load chosen sample into form
function loadSampleNotice(sampleId) {
  const sample = sampleNotices.find(s => s.id === sampleId);
  if (!sample) return;

  document.getElementById("notice-title").value = sample.title;
  document.getElementById("notice-text").value = sample.text;
}

// Handle Notice Analysis submission
async function handleFormSubmit(e) {
  e.preventDefault();

  const rawText = document.getElementById("notice-text").value.trim();
  if (!rawText) {
    alert("Please enter or paste raw notice text.");
    return;
  }

  const payload = {
    raw_text: rawText,
    notice_title: document.getElementById("notice-title").value,
    user_profile: {
      entity_name: document.getElementById("entity-name").value,
      entity_type: document.getElementById("entity-type").value,
      turnover_lakhs: parseFloat(document.getElementById("turnover-lakhs").value) || 120.0,
      industry_sector: document.getElementById("industry-sector").value,
      tax_registered: document.getElementById("tax-registered").value === "true",
      employee_count: parseInt(document.getElementById("employee-count").value) || 25
    }
  };

  const btn = e.target.querySelector("button[type='submit']");
  const origBtnText = btn.innerHTML;
  btn.innerHTML = "<span>⏳ Analyzing with TrustExtract-N...</span>";
  btn.disabled = true;

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json();
      alert(`Error: ${errData.detail || "Failed to analyze notice"}`);
      return;
    }

    currentAnalysisResult = await res.json();
    renderAnalysisResults(currentAnalysisResult);

  } catch (err) {
    alert("API Request Failed: " + err.message);
  } finally {
    btn.innerHTML = origBtnText;
    btn.disabled = false;
  }
}

// Render complete analysis output in UI
function renderAnalysisResults(data) {
  document.getElementById("results-placeholder").classList.add("hidden");
  document.getElementById("results-container").classList.remove("hidden");

  // Notice ID & Authority
  document.getElementById("notice-id-badge").innerText = data.notice_id;
  document.getElementById("authority-text").innerText = `${data.metadata.issuing_authority} (${data.metadata.notice_ref_no})`;

  // Trust Score & Metrics
  const tm = data.trust_metrics;
  document.getElementById("trust-score-val").innerText = tm.overall_trust_score.toFixed(1);
  document.getElementById("trust-confidence-badge").innerText = tm.confidence_level;

  document.getElementById("score-grounding").innerText = `${tm.grounding_score}%`;
  document.getElementById("bar-grounding").style.width = `${tm.grounding_score}%`;

  document.getElementById("score-format").innerText = `${tm.format_authenticity}%`;
  document.getElementById("bar-format").style.width = `${tm.format_authenticity}%`;

  document.getElementById("score-authority").innerText = `${tm.authority_verification}%`;
  document.getElementById("bar-authority").style.width = `${tm.authority_verification}%`;

  document.getElementById("score-rule").innerText = `${tm.rule_consistency}%`;
  document.getElementById("bar-rule").style.width = `${tm.rule_consistency}%`;

  // Applicability Banner
  const appBanner = document.getElementById("applicability-banner");
  const appStatus = document.getElementById("applicability-status");
  const appReason = document.getElementById("applicability-reason");

  if (data.user_applicability.is_applicable) {
    appBanner.className = "applicability-box applicable";
    appStatus.innerText = "✅ Applicable Notice";
    appReason.innerText = data.user_applicability.reason;
  } else {
    appBanner.className = "applicability-box exempt";
    appStatus.innerText = "🚫 Non-Applicable / Exempt";
    appReason.innerText = data.user_applicability.reason;
  }

  // Obligations List
  const obContainer = document.getElementById("obligations-list");
  obContainer.innerHTML = "";

  data.obligations.forEach(ob => {
    const pClass = ob.priority.toLowerCase();
    const card = document.createElement("div");
    card.className = "obligation-card";
    card.innerHTML = `
      <div class="ob-header">
        <div class="ob-title">${ob.title}</div>
        <span class="priority-tag ${pClass}">${ob.priority}</span>
      </div>
      <div class="ob-desc">${ob.description}</div>
      <div class="ob-meta-grid">
        <div class="ob-meta-item">
          <label>Target Deadline</label>
          <span>${ob.calculated_deadline} (${ob.days_remaining} days left)</span>
        </div>
        <div class="ob-meta-item">
          <label>Penalty Clause</label>
          <span>${ob.penalty_clause || "Statutory fines"}</span>
        </div>
        <div class="ob-meta-item">
          <label>Status</label>
          <span style="color:var(--accent-amber);">${ob.status}</span>
        </div>
      </div>
      <div class="evidence-highlight">
        <strong style="color:#fff;">Evidence Grounding Span [Offset ${ob.start_offset}-${ob.end_offset}]:</strong> "${ob.evidence_snippet}"
      </div>
    `;
    obContainer.appendChild(card);
  });

  // Evidence Audit List
  const evContainer = document.getElementById("evidence-list");
  evContainer.innerHTML = "";

  data.evidence_groundings.forEach(eg => {
    const item = document.createElement("div");
    item.className = "obligation-card";
    item.innerHTML = `
      <div class="flex-between mb-2">
        <strong style="color:var(--accent-cyan);">${eg.field_name} (Confidence: ${(eg.confidence * 100).toFixed(0)}%)</strong>
        <span style="font-family:var(--font-code); font-size:0.75rem; color:var(--text-muted);">Span: ${eg.start_offset}..${eg.end_offset}</span>
      </div>
      <div style="font-family:var(--font-code); font-size:0.85rem; color:#e5e7eb;">
        "${eg.snippet}"
      </div>
    `;
    evContainer.appendChild(item);
  });

  // Action Roadmap Steps
  const actContainer = document.getElementById("action-steps-list");
  actContainer.innerHTML = "";

  data.action_steps.forEach(step => {
    const sCard = document.createElement("div");
    sCard.className = "obligation-card";
    sCard.innerHTML = `
      <div class="ob-header">
        <div class="ob-title">Step ${step.step_number}: ${step.action_title}</div>
        <span class="priority-tag medium">${step.recommended_deadline}</span>
      </div>
      <div class="ob-desc">${step.description}</div>
      <div class="ob-meta-grid">
        <div class="ob-meta-item">
          <label>Assigned Lead</label>
          <span>${step.assigned_role}</span>
        </div>
        <div class="ob-meta-item">
          <label>Key Deliverable</label>
          <span>${step.deliverable}</span>
        </div>
      </div>
    `;
    actContainer.appendChild(sCard);
  });

  // Response Draft
  document.getElementById("response-draft-text").innerText = data.compliance_response_draft;
}

// Tab Switching
function switchTab(tabId) {
  document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
  document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));

  event.target.classList.add("active");
  const target = document.getElementById(tabId);
  if (target) target.classList.add("active");
}

// Copy Response Draft
function copyResponseDraft() {
  const text = document.getElementById("response-draft-text").innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert("Compliance Response Draft copied to clipboard!");
  });
}
