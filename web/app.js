/**
 * SynapseResearch — Frontend Application Controller
 * Handles tabs, real-time SSE streaming, interactive checkpoints,
 * in-app text highlight annotations, and local model supervision.
 */

// Global Application State
const AppState = {
  currentTab: 'tab-new',
  activeRunId: null,
  activeRunSummary: null,
  activeReportVersion: 1,
  annotations: [],
  sources: [],
  eventSource: null,
  settings: {},
  discoveredModels: [],
  selectedTextQuote: '',
  selectedTextRange: null,
  selectedIntent: 'auto',
  serverPollingInterval: null,
};

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initFormControls();
  initSettings();
  initLocalModels();
  initReportAnnotations();
  initHistory();
  loadInitialData();
});

/* ==========================================================================
   Navigation
   ========================================================================== */

function initNavigation() {
  const navButtons = document.querySelectorAll('.nav-item');
  navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      switchTab(targetTab);
    });
  });
}

function switchTab(tabId) {
  AppState.currentTab = tabId;

  // Update nav buttons
  document.querySelectorAll('.nav-item').forEach(b => {
    b.classList.toggle('active', b.getAttribute('data-tab') === tabId);
  });

  // Update tab panes
  document.querySelectorAll('.tab-pane').forEach(p => {
    p.classList.toggle('active', p.id === tabId);
  });

  // Topbar title update
  const titles = {
    'tab-new': ['Initiate Research', 'Collaborative agents with web grounding and iterative checkpoints'],
    'tab-run': ['Active Research Execution', 'Real-time telemetry, agent squad activities, and steering checkpoints'],
    'tab-report': ['Research Report & Review', 'Interactive reading, passage annotations, and revision orchestration'],
    'tab-local': ['Local Models & Server', 'Scan directories for GGUF weights, inspect fit, and manage llama.cpp'],
    'tab-settings': ['System Settings', 'API keys, search backend fallback order, and model overrides'],
    'tab-history': ['Research History', 'Review, export, or continue previous research runs'],
  };

  const [heading, subheading] = titles[tabId] || ['SynapseResearch', ''];
  document.getElementById('topbar-heading').textContent = heading;
  document.getElementById('topbar-subheading').textContent = subheading;

  // Tab specific refreshes
  if (tabId === 'tab-local') {
    fetchLocalModelsData();
  } else if (tabId === 'tab-history') {
    fetchHistoryRuns();
  }
}

/* ==========================================================================
   Initial Data Loading
   ========================================================================== */

async function loadInitialData() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    AppState.settings = data.settings;

    // Hardware Pill
    if (data.hardware) {
      const ram = data.hardware.ram;
      document.getElementById('hw-ram-text').textContent = `RAM: ${ram.available_gb} / ${ram.total_gb} GB`;
      if (data.hardware.gpus && data.hardware.gpus.length > 0) {
        const gpu = data.hardware.gpus[0];
        document.getElementById('hw-gpu-text').textContent = `GPU: ${gpu.name} (${gpu.free_vram_gb} GB free)`;
      } else {
        document.getElementById('hw-gpu-text').textContent = 'GPU: None (CPU mode)';
      }
    }

    // Sidebar provider badge
    updateSidebarProvider(data.settings.provider, data.settings.openrouter_default_model);

    // Suggested folder chips
    if (data.suggested_directories) {
      renderSuggestedFolderChips(data.suggested_directories);
    }

    // llama-server status
    if (data.llama_server) {
      updateLlamaServerUI(data.llama_server);
    }

    // Populate Settings UI
    populateSettingsForm(data.settings);
  } catch (err) {
    console.error('Failed to load initial settings:', err);
  }
}

function updateSidebarProvider(provider, model) {
  const nameEl = document.getElementById('sidebar-provider-name');
  const modelEl = document.getElementById('sidebar-provider-model');
  if (provider === 'llamacpp') {
    nameEl.textContent = 'Local llama.cpp';
    modelEl.textContent = AppState.settings.llamacpp_model_alias || 'Active GGUF';
  } else {
    nameEl.textContent = 'OpenRouter';
    modelEl.textContent = model || 'meta-llama/llama-3.3-70b';
  }
}

/* ==========================================================================
   Tab 1: New Research
   ========================================================================== */

function initFormControls() {
  // Toggle extra instructions
  const toggleBtn = document.getElementById('toggle-extra-btn');
  const extraWrapper = document.getElementById('extra-wrapper');
  toggleBtn.addEventListener('click', () => {
    extraWrapper.classList.toggle('hidden');
  });

  // Segmented selectors (Depth & Checkpoints)
  setupSegmentedControl('depth-selector');
  setupSegmentedControl('checkpoint-selector');

  // Start research button
  const startBtn = document.getElementById('btn-start-research');
  startBtn.addEventListener('click', handleStartResearch);
}

function setupSegmentedControl(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const buttons = container.querySelectorAll('.segment-btn');
  buttons.forEach(btn => {
    btn.addEventListener('click', () => {
      buttons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
}

function getSegmentedValue(containerId) {
  const container = document.getElementById(containerId);
  const active = container.querySelector('.segment-btn.active');
  return active ? active.getAttribute('data-value') : null;
}

async function handleStartResearch() {
  const topic = document.getElementById('input-topic').value.trim();
  if (!topic) {
    alert('Please enter a research topic or question.');
    document.getElementById('input-topic').focus();
    return;
  }

  const extra = document.getElementById('input-extra').value.trim();
  const depth = getSegmentedValue('depth-selector') || 'standard';
  const checkpointMode = getSegmentedValue('checkpoint-selector') || 'all';

  const startBtn = document.getElementById('btn-start-research');
  startBtn.disabled = true;
  startBtn.innerHTML = 'Initiating Agents...';

  try {
    const res = await fetch('/api/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic: topic,
        extra_instructions: extra,
        depth: depth,
        checkpoint_mode: checkpointMode,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Failed to start research run');
    }

    const runSummary = await res.json();
    AppState.activeRunId = runSummary.run_id;
    AppState.activeRunSummary = runSummary;

    // Reset console and agent list
    document.getElementById('console-log-stream').innerHTML = '';
    document.getElementById('agent-cards-list').innerHTML = '';
    document.getElementById('active-run-indicator').classList.remove('hidden');

    // Switch to active run tab
    switchTab('tab-run');

    // Connect SSE stream
    connectRunSSE(runSummary.run_id);
  } catch (err) {
    alert(`Error starting research: ${err.message}`);
  } finally {
    startBtn.disabled = false;
    startBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
      Launch Research Agents
    `;
  }
}

/* ==========================================================================
   Tab 2: Active Run & SSE Streaming
   ========================================================================== */

function connectRunSSE(runId) {
  if (AppState.eventSource) {
    AppState.eventSource.close();
  }

  const es = new EventSource(`/api/runs/${runId}/events`);
  AppState.eventSource = es;

  es.addEventListener('agent_event', (e) => {
    try {
      const event = JSON.parse(e.data);
      handleAgentEvent(event);
    } catch (err) {
      console.error('Failed to parse SSE agent event:', err);
    }
  });

  es.onerror = (e) => {
    console.log('SSE connection status update:', e);
  };

  // Cancel Run Button
  document.getElementById('btn-cancel-run').onclick = async () => {
    if (confirm('Are you sure you want to cancel the current research run?')) {
      await fetch(`/api/runs/${runId}/cancel`, { method: 'POST' });
    }
  };
}

function handleAgentEvent(event) {
  // 1. Append to console log
  appendConsoleLog(event);

  // 2. Update stage tracker steps
  updateStageTracker(event.stage);

  // 3. Update agent card
  updateAgentCard(event);

  // 4. Checkpoint check
  if (event.stage.startsWith('awaiting_') || event.message.includes('paused at Checkpoint')) {
    triggerCheckpointUI(event);
  }

  // 5. Completion check
  if (event.stage === 'done') {
    document.getElementById('active-run-indicator').classList.add('hidden');
    loadReportIntoView(AppState.activeRunId, event.data?.version || 1);
  }
}

function appendConsoleLog(event) {
  const consoleEl = document.getElementById('console-log-stream');
  const entry = document.createElement('div');
  entry.className = `log-entry ${event.status || ''}`;

  const timeStr = new Date(event.timestamp * 1000).toLocaleTimeString();
  entry.innerHTML = `
    <span class="log-time">[${timeStr}]</span>
    <strong>[${event.agent_name || event.agent_id}]:</strong> ${escapeHtml(event.message)}
  `;

  consoleEl.appendChild(entry);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function updateStageTracker(stage) {
  const steps = {
    'planning': 'step-planning',
    'awaiting_plan_review': 'step-cp1',
    'researching': 'step-researching',
    'awaiting_sections_review': 'step-cp2',
    'synthesizing': 'step-synthesizing',
    'reviewing': 'step-synthesizing',
    'awaiting_draft_review': 'step-cp3',
    'done': 'step-done',
  };

  const currentStepId = steps[stage];
  if (!currentStepId) return;

  const allStepIds = [
    'step-planning', 'step-cp1', 'step-researching',
    'step-cp2', 'step-synthesizing', 'step-cp3', 'step-done'
  ];

  const currentIndex = allStepIds.indexOf(currentStepId);

  allStepIds.forEach((id, idx) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (idx < currentIndex) {
      el.classList.add('completed');
    } else if (idx === currentIndex) {
      el.classList.add('active');
    }
  });
}

function updateAgentCard(event) {
  if (event.agent_id === 'system') return;

  const container = document.getElementById('agent-cards-list');
  let card = document.getElementById(`agent-card-${event.agent_id}`);

  if (!card) {
    // Remove empty placeholder
    const empty = container.querySelector('.empty-state-card');
    if (empty) empty.remove();

    card = document.createElement('div');
    card.id = `agent-card-${event.agent_id}`;
    card.className = 'agent-card';
    container.appendChild(card);
  }

  card.className = `agent-card ${event.status === 'completed' ? 'completed' : 'running'}`;
  card.innerHTML = `
    <div class="agent-card-info">
      <span class="agent-name-text">${escapeHtml(event.agent_name)}</span>
      <span class="agent-status-msg">${escapeHtml(event.message)}</span>
    </div>
    <span class="agent-badge-pill ${event.status}">${event.status}</span>
  `;

  // Update total agent badge count
  const count = container.querySelectorAll('.agent-card').length;
  document.getElementById('agent-count-badge').textContent = `${count} Agents`;
}

/* ==========================================================================
   Interactive Checkpoints
   ========================================================================== */

async function triggerCheckpointUI(event) {
  const card = document.getElementById('checkpoint-action-card');
  const headline = document.getElementById('cp-headline');
  const desc = document.getElementById('cp-description');
  const contentBox = document.getElementById('cp-content-box');
  card.classList.remove('hidden');

  // Fetch full run details to render structured payload
  const res = await fetch(`/api/runs/${AppState.activeRunId}`);
  const runData = await res.json();

  let stage = 'plan';
  if (event.stage === 'awaiting_sections_review' || event.message.includes('sections')) {
    stage = 'sections';
  } else if (event.stage === 'awaiting_draft_review' || event.message.includes('draft')) {
    stage = 'draft';
  }

  if (stage === 'plan') {
    headline.textContent = 'Checkpoint 1: Research Plan Review';
    desc.textContent = 'Review the proposed sub-topics and scoping questions. Approve or suggest corrections.';

    const plan = runData.plan;
    if (plan) {
      let subtasksHtml = plan.subtasks.map((st, i) => `
        <div style="margin-bottom: 12px; padding: 10px; background: rgba(255,255,255,0.04); border-radius: 6px;">
          <strong>#${i + 1} ${escapeHtml(st.title)}</strong> (Priority: ${st.priority})
          <p style="font-size: 0.8rem; color: #94a3b8; margin: 4px 0;">${escapeHtml(st.objective)}</p>
          <div style="font-size: 0.75rem; color: #64748b;">Key Questions: ${escapeHtml(st.key_questions.join(' • '))}</div>
        </div>
      `).join('');

      contentBox.innerHTML = `
        <div style="margin-bottom: 10px;">
          <strong>Interpretation:</strong> ${escapeHtml(plan.interpretation || '')}
        </div>
        <div><strong>Sub-Tasks:</strong></div>
        ${subtasksHtml}
      `;
    }
  } else if (stage === 'sections') {
    headline.textContent = 'Checkpoint 2: Section Reports Review';
    desc.textContent = 'Researchers have completed individual section reports. Review the findings before synthesis.';

    const sections = runData.sections || [];
    let secHtml = sections.map(sec => `
      <div style="margin-bottom: 10px; padding: 10px; background: rgba(255,255,255,0.04); border-radius: 6px;">
        <strong>${escapeHtml(sec.title)}</strong> — <em>Confidence: ${sec.confidence}</em> (${sec.sources.length} sources)
        <div style="font-size: 0.78rem; color: #cbd5e1; margin-top: 4px;">${escapeHtml(sec.markdown.slice(0, 200))}...</div>
      </div>
    `).join('');
    contentBox.innerHTML = secHtml || '<p>Sections ready for synthesis.</p>';
  } else if (stage === 'draft') {
    headline.textContent = 'Checkpoint 3: Synthesized Draft Review';
    desc.textContent = 'Examine the combined report draft and reviewer critique before finalization.';
    contentBox.innerHTML = `
      <div style="max-height: 200px; overflow-y: auto; font-size: 0.82rem; white-space: pre-wrap;">
        ${escapeHtml((runData.report || '').slice(0, 1500))}...
      </div>
    `;
  }

  // Setup buttons
  const approveBtn = document.getElementById('cp-btn-approve');
  const feedbackBtn = document.getElementById('cp-btn-feedback');
  const feedbackInput = document.getElementById('cp-feedback-input');

  approveBtn.onclick = async () => {
    approveBtn.disabled = true;
    await submitCheckpoint(stage, 'approve');
    card.classList.add('hidden');
    approveBtn.disabled = false;
  };

  feedbackBtn.onclick = async () => {
    const text = feedbackInput.value.trim();
    if (!text) {
      alert('Please enter steering feedback or instructions before clicking Send Feedback.');
      feedbackInput.focus();
      return;
    }
    feedbackBtn.disabled = true;
    await submitCheckpoint(stage, 'feedback', text);
    feedbackInput.value = '';
    card.classList.add('hidden');
    feedbackBtn.disabled = false;
  };
}

async function submitCheckpoint(stage, actionType, feedbackText = null) {
  try {
    await fetch(`/api/runs/${AppState.activeRunId}/checkpoint`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        stage: stage,
        action: actionType,
        feedback_text: feedbackText,
      }),
    });
  } catch (err) {
    alert(`Failed to submit checkpoint response: ${err.message}`);
  }
}

/* ==========================================================================
   Tab 3: Report & In-App Annotations
   ========================================================================== */

async function loadReportIntoView(runId, version = 1) {
  AppState.activeRunId = runId;
  AppState.activeReportVersion = version;

  try {
    const res = await fetch(`/api/runs/${runId}/report?version=${version}`);
    if (!res.ok) throw new Error('Report not found');
    const data = await res.json();

    // Populate version select
    const versionSelect = document.getElementById('report-version-select');
    versionSelect.innerHTML = '';
    (data.versions || [1]).forEach(v => {
      const opt = document.createElement('option');
      opt.value = v;
      opt.textContent = `v${v} ${v === 1 ? '(Initial)' : '(Revised)'}`;
      if (v === version) opt.selected = true;
      versionSelect.appendChild(opt);
    });

    versionSelect.onchange = () => {
      loadReportIntoView(runId, parseInt(versionSelect.value));
    };

    // Render markdown
    renderReportMarkdown(data.markdown);

    // Load annotations for this version
    await loadAnnotations(runId, version);

    // Switch to report tab
    switchTab('tab-report');
  } catch (err) {
    console.error('Failed to load report:', err);
  }
}

function renderReportMarkdown(rawMarkdown) {
  const container = document.getElementById('report-content-body');
  if (!rawMarkdown) {
    container.innerHTML = '<div class="empty-state-card">Report is empty.</div>';
    return;
  }

  // Parse markdown
  const html = marked.parse(rawMarkdown);
  const cleanHtml = DOMPurify.sanitize(html);

  // Post-process to turn [S1], [S2] into interactive citation pills
  const citationReplaced = cleanHtml.replace(
    /\[(S\d+)\]/g,
    '<span class="citation-ref" data-source-id="$1">[$1]</span>'
  );

  container.innerHTML = citationReplaced;

  // Add click handlers for citations
  container.querySelectorAll('.citation-ref').forEach(pill => {
    pill.addEventListener('click', (e) => {
      const sId = pill.getAttribute('data-source-id');
      openSourceModal(sId);
    });
  });

  // Extract source count from references
  const refsMatch = rawMarkdown.match(/\[S\d+\]/g) || [];
  const uniqueRefs = new Set(refsMatch);
  document.getElementById('report-sources-count').textContent = `${uniqueRefs.size} Sources Cited`;
}

function initReportAnnotations() {
  const reportBody = document.getElementById('report-content-body');
  const popover = document.getElementById('selection-popover');
  const commentBtn = document.getElementById('popover-btn-comment');

  // Mouseup listener for text selection
  reportBody.addEventListener('mouseup', (e) => {
    const selection = window.getSelection();
    const text = selection.toString().trim();

    if (text.length > 3) {
      AppState.selectedTextQuote = text;
      // Position popover near mouse
      popover.style.left = `${e.pageX + 10}px`;
      popover.style.top = `${e.pageY - 35}px`;
      popover.classList.remove('hidden');
    } else {
      popover.classList.add('hidden');
    }
  });

  // Open modal on click
  commentBtn.addEventListener('click', () => {
    popover.classList.add('hidden');
    openAnnotationModal(AppState.selectedTextQuote);
  });

  // Modal controls
  document.getElementById('modal-close-btn').onclick = closeAnnotationModal;
  document.getElementById('modal-cancel-btn').onclick = closeAnnotationModal;
  document.getElementById('modal-save-btn').onclick = saveAnnotation;

  // Intent chips
  document.querySelectorAll('#modal-intent-chips .intent-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('#modal-intent-chips .intent-chip').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      AppState.selectedIntent = chip.getAttribute('data-intent');
    });
  });

  // Submit revision button
  document.getElementById('btn-submit-feedback').onclick = triggerFeedbackRevision;

  // Export buttons
  document.getElementById('btn-export-md').onclick = () => exportReport('md');
  document.getElementById('btn-export-html').onclick = () => exportReport('html');
  document.getElementById('btn-copy-report').onclick = copyReportToClipboard;
}

function openAnnotationModal(quote) {
  document.getElementById('modal-quote-preview').textContent = `"${quote}"`;
  document.getElementById('modal-comment-input').value = '';
  document.getElementById('annotation-modal').classList.remove('hidden');
  document.getElementById('modal-comment-input').focus();
}

function closeAnnotationModal() {
  document.getElementById('annotation-modal').classList.add('hidden');
}

async function saveAnnotation() {
  const comment = document.getElementById('modal-comment-input').value.trim();
  if (!comment) {
    alert('Please write your comment or feedback instructions.');
    return;
  }

  const annotation = {
    id: Math.random().toString(36).substring(2, 9),
    report_version: AppState.activeReportVersion,
    quote: AppState.selectedTextQuote,
    comment: comment,
    intent: AppState.selectedIntent,
  };

  AppState.annotations.push(annotation);
  closeAnnotationModal();

  // Save to backend
  await fetch(`/api/runs/${AppState.activeRunId}/annotations`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      version: AppState.activeReportVersion,
      annotations: AppState.annotations,
    }),
  });

  renderAnnotationsDrawer();
}

async function loadAnnotations(runId, version) {
  try {
    const res = await fetch(`/api/runs/${runId}/annotations?version=${version}`);
    const data = await res.json();
    AppState.annotations = data.annotations || [];
    renderAnnotationsDrawer();
  } catch (err) {
    console.error('Failed to load annotations:', err);
  }
}

function renderAnnotationsDrawer() {
  const container = document.getElementById('annotations-list-items');
  const count = AppState.annotations.length;
  document.getElementById('annotation-count-title').textContent = `In-App Feedback (${count})`;

  const submitBtn = document.getElementById('btn-submit-feedback');
  submitBtn.disabled = count === 0;

  if (count === 0) {
    container.innerHTML = '<p class="empty-hint">Highlight any passage in the report to leave a comment or request deeper research.</p>';
    return;
  }

  container.innerHTML = AppState.annotations.map(a => `
    <div class="annotation-card">
      <div class="annotation-quote">"${escapeHtml(a.quote)}"</div>
      <div class="annotation-comment"><strong>[${a.intent}]:</strong> ${escapeHtml(a.comment)}</div>
    </div>
  `).join('');
}

async function triggerFeedbackRevision() {
  if (AppState.annotations.length === 0) return;

  const btn = document.getElementById('btn-submit-feedback');
  btn.disabled = true;
  btn.textContent = 'Launching Feedback Triage...';

  try {
    await fetch(`/api/runs/${AppState.activeRunId}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        version: AppState.activeReportVersion,
        annotations: AppState.annotations,
      }),
    });

    // Switch to active run tab to observe revision
    switchTab('tab-run');
    connectRunSSE(AppState.activeRunId);
  } catch (err) {
    alert(`Error submitting feedback: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = '🚀 Generate Revised Report';
  }
}

function openSourceModal(sourceId) {
  const modal = document.getElementById('source-detail-modal');
  document.getElementById('source-modal-id').textContent = sourceId;
  document.getElementById('source-modal-title').textContent = `Source [${sourceId}] Details`;
  document.getElementById('source-modal-snippet').textContent = `Fetching reference details for ${sourceId}...`;
  document.getElementById('source-modal-close').onclick = () => modal.classList.add('hidden');
  modal.classList.remove('hidden');

  // Find in run data
  fetch(`/api/runs/${AppState.activeRunId}`).then(r => r.json()).then(data => {
    let matchedSource = null;
    (data.sections || []).forEach(sec => {
      sec.sources.forEach(s => {
        if (s.id === sourceId) matchedSource = s;
      });
    });

    if (matchedSource) {
      document.getElementById('source-modal-title').textContent = matchedSource.title || 'Web Reference';
      document.getElementById('source-modal-backend').textContent = matchedSource.backend;
      document.getElementById('source-modal-link').href = matchedSource.url;
      document.getElementById('source-modal-snippet').textContent = matchedSource.snippet;
    }
  });
}

function exportReport(format) {
  const text = document.getElementById('report-content-body').innerText;
  const blob = new Blob([text], { type: format === 'md' ? 'text/markdown' : 'text/html' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `research_report_${AppState.activeRunId}_v${AppState.activeReportVersion}.${format}`;
  a.click();
}

function copyReportToClipboard() {
  const text = document.getElementById('report-content-body').innerText;
  navigator.clipboard.writeText(text).then(() => {
    alert('Report copied to clipboard!');
  });
}

/* ==========================================================================
   Tab 4: Local Models & llama-server
   ========================================================================== */

function initLocalModels() {
  // Folder picker button
  document.getElementById('btn-add-folder-picker').onclick = async () => {
    try {
      const res = await fetch('/api/local/folders/pick', { method: 'POST' });
      const data = await res.json();
      if (data.selected_path) {
        await addGrantedFolder(data.selected_path);
      }
    } catch (err) {
      alert(`Folder picker failed: ${err.message}`);
    }
  };

  // Rescan models button
  document.getElementById('btn-rescan-models').onclick = scanModels;

  // Stop server button
  document.getElementById('btn-server-stop').onclick = async () => {
    await fetch('/api/local/server/stop', { method: 'POST' });
    fetchLocalModelsData();
  };
}

async function fetchLocalModelsData() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();

    // Render granted folders
    renderGrantedFolders(data.settings.local_folders || []);

    // Update llama-server status
    if (data.llama_server) {
      updateLlamaServerUI(data.llama_server);
    }

    // Scan models
    await scanModels();
  } catch (err) {
    console.error('Failed to fetch local model data:', err);
  }
}

function renderSuggestedFolderChips(paths) {
  const container = document.getElementById('suggested-chips-container');
  container.innerHTML = paths.map(p => `
    <button class="chip-btn" type="button" data-path="${escapeHtml(p)}">
      + ${escapeHtml(p.split('\\').pop() || p)}
    </button>
  `).join('');

  container.querySelectorAll('.chip-btn').forEach(btn => {
    btn.onclick = () => addGrantedFolder(btn.getAttribute('data-path'));
  });
}

function renderGrantedFolders(folders) {
  const container = document.getElementById('granted-folders-list');
  if (folders.length === 0) {
    container.innerHTML = '<p class="empty-hint">No directories granted yet. Click "Browse & Add Folder" or choose a detected cache above.</p>';
    return;
  }

  container.innerHTML = folders.map(f => `
    <div class="folder-row">
      <span>📁 ${escapeHtml(f)}</span>
      <button class="btn btn-sm btn-outline" onclick="removeGrantedFolder('${escapeHtml(f)}')">Remove</button>
    </div>
  `).join('');
}

async function addGrantedFolder(folder) {
  await fetch('/api/local/folders', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder }),
  });
  fetchLocalModelsData();
}

window.removeGrantedFolder = async function(folder) {
  await fetch('/api/local/folders', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ folder }),
  });
  fetchLocalModelsData();
};

async function scanModels() {
  const tableBody = document.getElementById('models-table-body');
  tableBody.innerHTML = '<tr><td colspan="7" class="empty-table-cell">Scanning granted directories for models...</td></tr>';

  try {
    const res = await fetch('/api/local/scan', { method: 'POST' });
    const data = await res.json();
    AppState.discoveredModels = data.models || [];

    if (AppState.discoveredModels.length === 0) {
      tableBody.innerHTML = '<tr><td colspan="7" class="empty-table-cell">No models found in granted directories.</td></tr>';
      return;
    }

    tableBody.innerHTML = AppState.discoveredModels.map(m => {
      const fit = m.fit_recommendation || {};
      const fitBadgeClass = fit.status || 'moderate';
      return `
        <tr>
          <td>
            <strong>${escapeHtml(m.name || m.filename)}</strong>
            <div style="font-size: 0.72rem; color: #64748b;">${escapeHtml(m.path)}</div>
          </td>
          <td><span class="badge">${m.format}</span></td>
          <td>${m.size_formatted || '--'}</td>
          <td>${m.architecture || '--'}</td>
          <td>${m.quantization || '--'}</td>
          <td>
            <span class="fit-badge ${fitBadgeClass}">${fit.fit || 'Unknown'}</span>
          </td>
          <td>
            ${m.runnable ? `
              <button class="btn btn-sm btn-primary" onclick="launchModel('${escapeHtml(m.path)}')">
                Launch Server
              </button>
            ` : `<span style="font-size:0.72rem; color:#94a3b8;">${m.note || 'Not directly runnable'}</span>`}
          </td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="7" class="empty-table-cell">Scan failed: ${err.message}</td></tr>`;
  }
}

window.launchModel = async function(modelPath) {
  if (!confirm(`Launch llama-server with model:\n${modelPath}?`)) return;

  const btnStop = document.getElementById('btn-server-stop');
  btnStop.disabled = true;

  try {
    const res = await fetch('/api/local/server/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_path: modelPath }),
    });
    const data = await res.json();
    if (data.success) {
      alert('llama-server successfully launched and ready!');
      updateSidebarProvider('llamacpp', modelPath.split('\\').pop());
    } else {
      alert(`Server failed to start: ${data.status?.error_message || 'Unknown error'}`);
    }
  } catch (err) {
    alert(`Failed to start llama-server: ${err.message}`);
  } finally {
    fetchLocalModelsData();
  }
};

function updateLlamaServerUI(serverInfo) {
  const pulse = document.getElementById('llama-server-pulse');
  const urlText = document.getElementById('llama-server-url-text');
  const binPathText = document.getElementById('binary-path-text');
  const stopBtn = document.getElementById('btn-server-stop');

  if (serverInfo.found) {
    binPathText.textContent = `Binary: ${serverInfo.path} (${serverInfo.info?.version || 'Active'})`;
  } else {
    binPathText.textContent = 'llama-server executable not found on system PATH. Install via winget or configure in Settings.';
  }

  const isAlive = serverInfo.status?.is_alive;
  pulse.className = `status-pulse-badge ${isAlive ? 'running' : ''}`;
  urlText.textContent = isAlive
    ? `${serverInfo.status.base_url} (Running — PID: ${serverInfo.status.pid})`
    : 'http://127.0.0.1:8080/v1 (Stopped)';

  stopBtn.disabled = !isAlive;

  // Logs stream
  if (serverInfo.status?.recent_logs) {
    const logBox = document.getElementById('server-log-stream');
    logBox.innerHTML = serverInfo.status.recent_logs.map(l => `<div class="log-entry">${escapeHtml(l)}</div>`).join('');
    logBox.scrollTop = logBox.scrollHeight;
  }
}

/* ==========================================================================
   Tab 5: Settings
   ========================================================================== */

function initSettings() {
  // Provider radio toggle
  const radOpenRouter = document.getElementById('provider-choice-openrouter');
  const radLlamaCpp = document.getElementById('provider-choice-llamacpp');
  const fieldsOR = document.getElementById('fields-openrouter');
  const fieldsLC = document.getElementById('fields-llamacpp');

  radOpenRouter.addEventListener('change', () => {
    fieldsOR.classList.remove('hidden');
    fieldsLC.classList.add('hidden');
  });

  radLlamaCpp.addEventListener('change', () => {
    fieldsLC.classList.remove('hidden');
    fieldsOR.classList.add('hidden');
  });

  // Toggle API key visibility
  document.getElementById('btn-toggle-key-visibility').onclick = () => {
    const input = document.getElementById('setting-openrouter-key');
    const btn = document.getElementById('btn-toggle-key-visibility');
    if (input.type === 'password') {
      input.type = 'text';
      btn.textContent = 'Hide';
    } else {
      input.type = 'password';
      btn.textContent = 'Show';
    }
  };

  // Save settings
  document.getElementById('btn-save-settings').onclick = saveAllSettings;

  // Test provider connection
  document.getElementById('btn-test-provider').onclick = async () => {
    const btn = document.getElementById('btn-test-provider');
    btn.disabled = true;
    btn.textContent = 'Testing...';
    try {
      const res = await fetch('/api/provider/test', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        alert(`Connection Successful!\nProvider: ${data.provider}\nModels available: ${data.model_count}`);
      } else {
        alert(`Connection Failed:\n${data.error}`);
      }
    } catch (err) {
      alert(`Test request failed: ${err.message}`);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Test Connection';
    }
  };

  // Test search
  document.getElementById('btn-test-search').onclick = async () => {
    try {
      const res = await fetch('/api/search/test', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: 'quantum computing 2026' }),
      });
      const data = await res.json();
      alert(`Search test succeeded!\nFound ${data.results?.length} results across active backends.`);
    } catch (err) {
      alert(`Search test error: ${err.message}`);
    }
  };
}

function populateSettingsForm(s) {
  if (s.provider === 'llamacpp') {
    document.getElementById('provider-choice-llamacpp').checked = true;
    document.getElementById('fields-llamacpp').classList.remove('hidden');
    document.getElementById('fields-openrouter').classList.add('hidden');
  } else {
    document.getElementById('provider-choice-openrouter').checked = true;
    document.getElementById('fields-openrouter').classList.remove('hidden');
    document.getElementById('fields-llamacpp').classList.add('hidden');
  }

  if (s.has_openrouter_key) {
    document.getElementById('setting-openrouter-key').placeholder = s.openrouter_api_key_masked;
  }
  document.getElementById('setting-openrouter-model').value = s.openrouter_default_model || '';
  document.getElementById('setting-llamacpp-url').value = s.llamacpp_base_url || '';
  document.getElementById('setting-llamacpp-ctx').value = s.llama_server_ctx || 8192;
  document.getElementById('setting-llamacpp-ngl').value = s.llama_server_ngl || 99;
  document.getElementById('setting-llamacpp-parallel').value = s.llama_server_parallel || 2;

  document.getElementById('setting-search-order').value = (s.search_backends_order || []).join(', ');
  document.getElementById('setting-concurrency').value = s.max_researcher_concurrency || 2;
  document.getElementById('setting-max-reviews').value = s.max_review_rounds || 2;
  document.getElementById('setting-sources-per-sec').value = s.max_sources_per_section || 6;
}

async function saveAllSettings() {
  const provider = document.querySelector('input[name="provider-choice"]:checked').value;
  const orKey = document.getElementById('setting-openrouter-key').value.trim();
  const orModel = document.getElementById('setting-openrouter-model').value.trim();
  const lcUrl = document.getElementById('setting-llamacpp-url').value.trim();
  const lcCtx = parseInt(document.getElementById('setting-llamacpp-ctx').value);
  const lcNgl = parseInt(document.getElementById('setting-llamacpp-ngl').value);
  const lcParallel = parseInt(document.getElementById('setting-llamacpp-parallel').value);

  const searchOrder = document.getElementById('setting-search-order').value.split(',').map(s => s.trim()).filter(Boolean);
  const braveKey = document.getElementById('setting-brave-key').value.trim();
  const tavilyKey = document.getElementById('setting-tavily-key').value.trim();
  const serperKey = document.getElementById('setting-serper-key').value.trim();
  const searxngUrl = document.getElementById('setting-searxng-url').value.trim();

  const concurrency = parseInt(document.getElementById('setting-concurrency').value);
  const maxReviews = parseInt(document.getElementById('setting-max-reviews').value);
  const sourcesPerSec = parseInt(document.getElementById('setting-sources-per-sec').value);

  const payload = {
    provider,
    openrouter_api_key: orKey,
    openrouter_default_model: orModel,
    llamacpp_base_url: lcUrl,
    llama_server_ctx: lcCtx,
    llama_server_ngl: lcNgl,
    llama_server_parallel: lcParallel,
    search_backends_order: searchOrder,
    brave_api_key: braveKey,
    tavily_api_key: tavilyKey,
    serper_api_key: serperKey,
    searxng_url: searxngUrl,
    max_researcher_concurrency: concurrency,
    max_review_rounds: maxReviews,
    max_sources_per_section: sourcesPerSec,
  };

  try {
    const res = await fetch('/api/settings', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    AppState.settings = data.settings;
    updateSidebarProvider(provider, orModel);
    alert('Settings saved successfully!');
  } catch (err) {
    alert(`Failed to save settings: ${err.message}`);
  }
}

/* ==========================================================================
   Tab 6: History
   ========================================================================== */

function initHistory() {
  document.getElementById('btn-refresh-history').onclick = fetchHistoryRuns;
}

async function fetchHistoryRuns() {
  const container = document.getElementById('history-runs-grid');
  try {
    const res = await fetch('/api/runs');
    const data = await res.json();
    const runs = data.runs || [];

    if (runs.length === 0) {
      container.innerHTML = '<div class="empty-state-card">No historical runs recorded yet.</div>';
      return;
    }

    container.innerHTML = runs.map(r => {
      const dateStr = new Date(r.created_at * 1000).toLocaleString();
      return `
        <div class="history-card" onclick="openHistoricalRun('${r.run_id}', ${r.current_version || 1})">
          <div class="history-topic">${escapeHtml(r.topic)}</div>
          <div class="history-meta">
            <span>📅 ${dateStr}</span>
            <span>📊 v${r.current_version || 1}</span>
            <span>🌐 ${r.total_sources || 0} sources</span>
            <span class="badge">${r.state}</span>
          </div>
        </div>
      `;
    }).join('');
  } catch (err) {
    container.innerHTML = `<div class="empty-state-card">Failed to load history: ${err.message}</div>`;
  }
}

window.openHistoricalRun = function(runId, version) {
  loadReportIntoView(runId, version);
};

/* ==========================================================================
   Utility Helpers
   ========================================================================== */

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
