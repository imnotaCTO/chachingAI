const API_BASE = window.API_BASE || "http://127.0.0.1:8000";

const state = {
  data: null,
  eventId: null,
  sportsbook: "ALL",
  legs: [],
  loading: false,
  loadingProps: false,
  statsSource: "balldontlie",
  propsRequestId: 0,
  propsAbort: null,
  propsError: null,
  filters: {
    game: "",
    player: "",
    stat: "",
  },
};

const selectors = {
  sportsbookSelect: document.getElementById("sportsbookSelect"),
  statsSource: document.getElementById("statsSource"),
  dateInput: document.getElementById("dateInput"),
  todayBadge: document.getElementById("todayBadge"),
  lastUpdate: document.getElementById("lastUpdate"),
  gamesList: document.getElementById("gamesList"),
  eventTitle: document.getElementById("eventTitle"),
  propsList: document.getElementById("propsList"),
  legsList: document.getElementById("legsList"),
  parlayMeta: document.getElementById("parlayMeta"),
  modelProb: document.getElementById("modelProb"),
  impliedProb: document.getElementById("impliedProb"),
  fairOdds: document.getElementById("fairOdds"),
  expectedValue: document.getElementById("expectedValue"),
  sampleSize: document.getElementById("sampleSize"),
  gameFilter: document.getElementById("gameFilter"),
  playerFilter: document.getElementById("playerFilter"),
  statFilter: document.getElementById("statFilter"),
  toast: document.getElementById("toast"),
};

const formatPercent = (value) => {
  if (value === null || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
};

const formatOdds = (odds) => {
  if (odds === null || Number.isNaN(odds)) return "--";
  return odds > 0 ? `+${odds.toFixed(0)}` : `${odds.toFixed(0)}`;
};

const americanToProb = (odds) => {
  if (odds === 0) return null;
  if (odds > 0) return 100 / (odds + 100);
  return -odds / (-odds + 100);
};

const probToAmerican = (probability) => {
  if (probability <= 0 || probability >= 1) return null;
  if (probability >= 0.5) {
    return -100 * probability / (1 - probability);
  }
  return 100 * (1 - probability) / probability;
};

const expectedValue = (probability, odds) => {
  if (probability === null || odds === null) return null;
  const payout = odds > 0 ? odds / 100 : 100 / Math.abs(odds);
  return probability * payout - (1 - probability);
};

const showToast = (message, tone = "success") => {
  if (!selectors.toast) return;
  selectors.toast.textContent = message;
  selectors.toast.className = `toast show ${tone}`;
  window.clearTimeout(showToast._timer);
  showToast._timer = window.setTimeout(() => {
    selectors.toast.className = "toast";
  }, 1800);
};

const init = async () => {
  const today = new Date().toISOString().slice(0, 10);
  selectors.dateInput.value = today;
  updateTodayBadge(today);
  if (selectors.statsSource) {
    selectors.statsSource.value = state.statsSource;
  }
  bindEvents();
  await loadSportsbooks();
  await loadEvents(today);
  renderParlay();
};

const bindEvents = () => {
  selectors.sportsbookSelect.addEventListener("change", (event) => {
    state.sportsbook = event.target.value;
    state.legs = [];
    if (state.eventId) {
      loadProps(state.eventId);
    }
    renderParlay();
  });

  selectors.statsSource.addEventListener("change", (event) => {
    state.statsSource = event.target.value;
    state.legs = [];
    if (state.eventId) {
      loadProps(state.eventId);
    }
    renderParlay();
  });

  selectors.dateInput.addEventListener("change", (event) => {
    const date = event.target.value;
    if (date) {
      selectors.playerFilter.value = "";
      selectors.statFilter.value = "";
      state.filters.player = "";
      state.filters.stat = "";
      updateTodayBadge(date);
      loadEvents(date);
    }
  });

  selectors.gameFilter.addEventListener("input", (event) => {
    state.filters.game = event.target.value.toLowerCase();
    renderGames();
  });

  selectors.playerFilter.addEventListener("input", (event) => {
    state.filters.player = event.target.value.toLowerCase();
    renderProps();
  });

  selectors.statFilter.addEventListener("change", (event) => {
    state.filters.stat = event.target.value;
    renderProps();
  });
};

const fetchJson = async (path, { signal } = {}) => {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
};

const loadSportsbooks = async () => {
  try {
    const data = await fetchJson("/api/sportsbooks");
    state.data = state.data || {};
    state.data.sportsbooks = data.sportsbooks || [];
    if (!state.sportsbook) {
      state.sportsbook = "ALL";
    }
    renderSportsbooks();
  } catch (error) {
    showToast("Failed to load sportsbooks", "error");
  }
};

const updateTodayBadge = (dateValue) => {
  if (!selectors.todayBadge) return;
  const today = new Date().toISOString().slice(0, 10);
  if (dateValue === today) {
    selectors.todayBadge.classList.add("show");
  } else {
    selectors.todayBadge.classList.remove("show");
  }
};

const loadEvents = async (date) => {
  state.loading = true;
  try {
    const data = await fetchJson(`/api/events?date=${date}`);
    state.data = state.data || {};
    state.data.events = data.events || [];
    state.eventId = state.data.events[0]?.event_id || null;
    state.data.props = [];
    renderGames();
    if (state.eventId) {
      await loadProps(state.eventId);
    } else {
      selectors.propsList.innerHTML = `<div class="subtle">No events found for this date.</div>`;
    }
  } catch (error) {
    showToast("Failed to load events", "error");
  } finally {
    state.loading = false;
  }
};

const loadProps = async (eventId) => {
  state.loadingProps = true;
  state.propsError = null;
  renderProps();
  state.propsRequestId += 1;
  const requestId = state.propsRequestId;
  if (state.propsAbort) {
    state.propsAbort.abort();
  }
  const controller = new AbortController();
  state.propsAbort = controller;
  const timeout = window.setTimeout(() => controller.abort(), 12000);
  try {
    const sportsbookParam =
      state.sportsbook && state.sportsbook !== "ALL"
        ? `sportsbook=${encodeURIComponent(state.sportsbook)}&`
        : "";
    const statsSource = encodeURIComponent(state.statsSource || "balldontlie");
    const data = await fetchJson(
      `/api/events/${eventId}/props?${sportsbookParam}stats_source=${statsSource}&max_players=8`,
      { signal: controller.signal }
    );
    if (requestId !== state.propsRequestId) {
      return;
    }
    state.data = state.data || {};
    state.data.props = data.props || [];
    selectors.lastUpdate.textContent = data.last_update || "--";
    renderProps();
  } catch (error) {
    if (requestId !== state.propsRequestId) {
      return;
    }
    state.propsError = "Unable to load props.";
    showToast("Failed to load props", "error");
  } finally {
    window.clearTimeout(timeout);
    if (requestId === state.propsRequestId) {
      state.loadingProps = false;
      renderProps();
      if (state.propsAbort === controller) {
        state.propsAbort = null;
      }
    }
  }
};

const renderSportsbooks = () => {
  selectors.sportsbookSelect.innerHTML = "";
  const allOption = document.createElement("option");
  allOption.value = "ALL";
  allOption.textContent = "All Sportsbooks";
  if (state.sportsbook === "ALL") allOption.selected = true;
  selectors.sportsbookSelect.appendChild(allOption);

  (state.data?.sportsbooks || []).forEach((book) => {
    const option = document.createElement("option");
    option.value = book;
    option.textContent = book;
    if (book === state.sportsbook) option.selected = true;
    selectors.sportsbookSelect.appendChild(option);
  });
};

const renderGames = () => {
  selectors.gamesList.innerHTML = "";
  const filtered = (state.data?.events || []).filter((event) => {
    const query = state.filters.game;
    if (!query) return true;
    return `${event.away_team} ${event.home_team}`.toLowerCase().includes(query);
  });

  filtered.forEach((event) => {
    const card = document.createElement("div");
    card.className = "game-card";
    if (event.event_id === state.eventId) card.classList.add("active");
    card.innerHTML = `
      <strong>${event.away_team} @ ${event.home_team}</strong>
      <div class="meta">${new Date(event.commence_time).toLocaleString()}</div>
    `;
    card.addEventListener("click", () => {
      state.eventId = event.event_id;
      state.legs = [];
      renderGames();
      loadProps(event.event_id);
      renderParlay();
    });
    selectors.gamesList.appendChild(card);
  });
};

const renderProps = () => {
  selectors.propsList.innerHTML = "";
  const event = state.data?.events?.find((item) => item.event_id === state.eventId);
  selectors.eventTitle.textContent = event
    ? `${event.away_team} @ ${event.home_team}`
    : "Select a game";
  if (state.loadingProps) {
    selectors.propsList.innerHTML = `<div class="subtle">Loading props...</div>`;
    return;
  }
  if (state.propsError) {
    selectors.propsList.innerHTML = `<div class="subtle">${state.propsError}</div>`;
    return;
  }

  const filteredProps = (state.data?.props || []).filter((prop) => {
    if (state.eventId && prop.event_id !== state.eventId) return false;
    if (state.filters.player && !prop.player_name.toLowerCase().includes(state.filters.player)) {
      return false;
    }
    if (state.filters.stat && prop.stat !== state.filters.stat) return false;
    return true;
  });

  const grouped = new Map();
  filteredProps.forEach((prop) => {
    const key = `${prop.player_id || prop.player_name}-${prop.stat}`;
    if (!grouped.has(key)) {
      grouped.set(key, {
        player_name: prop.player_name,
        stat: prop.stat,
        items: [],
      });
    }
    grouped.get(key).items.push(prop);
  });

  if (grouped.size === 0) {
    const hasFilters = Boolean(state.filters.player || state.filters.stat);
    if ((state.data?.props || []).length === 0) {
      selectors.propsList.innerHTML = `<div class="subtle">No props returned from API for this event.</div>`;
    } else if (hasFilters) {
      selectors.propsList.innerHTML = `<div class="subtle">No props match the filters.</div>`;
    } else {
      selectors.propsList.innerHTML = `<div class="subtle">No props available for this event.</div>`;
    }
    return;
  }

  Array.from(grouped.values()).forEach((group) => {
    const row = document.createElement("div");
    row.className = "prop-row";
    row.innerHTML = `
      <div>
        <div class="player">${group.player_name}</div>
        <div class="meta">${group.stat.toUpperCase()}</div>
      </div>
      <div class="prop-options"></div>
    `;
    const options = row.querySelector(".prop-options");
    group.items
      .sort((a, b) => a.line - b.line)
      .forEach((prop) => {
        const option = document.createElement("button");
        option.type = "button";
        option.className = "prop-option";
        const evValue = expectedValue(prop.model_probability, prop.odds);
        const key = `${prop.event_id}-${prop.player_id || prop.player_name}-${prop.stat}`;
        const selected = state.legs.some((leg) => {
          const legKey = `${leg.event_id}-${leg.player_id || leg.player_name}-${leg.stat}`;
          return (
            legKey === key &&
            leg.line === prop.line &&
            leg.direction === prop.direction
          );
        });
        if (selected) option.classList.add("selected");
        option.innerHTML = `
          <div class="line">${prop.direction} ${prop.line}</div>
          <div class="odds">${formatOdds(prop.odds)}</div>
          <div class="meta">${prop.sportsbook || "Sportsbook"}</div>
          <div class="ev">${evValue === null ? "--" : `${(evValue * 100).toFixed(1)}% EV`}</div>
        `;
        option.addEventListener("click", () => addLeg(prop));
        options.appendChild(option);
      });
    selectors.propsList.appendChild(row);
  });
};

const addLeg = (prop) => {
  const key = `${prop.event_id}-${prop.player_id || prop.player_name}-${prop.stat}`;
  const existingIndex = state.legs.findIndex(
    (leg) => `${leg.event_id}-${leg.player_id || leg.player_name}-${leg.stat}` === key
  );
  if (existingIndex >= 0) {
    const existing = state.legs[existingIndex];
    if (
      existing.line === prop.line &&
      existing.direction === prop.direction &&
      existing.player_name === prop.player_name
    ) {
      return;
    }
    state.legs.splice(existingIndex, 1, { ...prop });
    showToast(`Replaced ${existing.player_name} ${existing.stat.toUpperCase()} leg`);
  } else {
    state.legs.push({ ...prop });
  }
  renderParlay();
  renderProps();
};

const removeLeg = (index) => {
  state.legs.splice(index, 1);
  renderParlay();
};

const renderParlay = () => {
  selectors.legsList.innerHTML = "";
  selectors.parlayMeta.textContent = `${state.legs.length} legs selected`;

  state.legs.forEach((leg, index) => {
    const card = document.createElement("div");
    card.className = "leg-card";
    card.innerHTML = `
      <div><strong>${leg.player_name}</strong> · ${leg.stat.toUpperCase()}</div>
      <div class="line">${leg.direction} ${leg.line} (${formatOdds(leg.odds)})</div>
      <button class="remove">Remove</button>
    `;
    card.querySelector(".remove").addEventListener("click", () => removeLeg(index));
    selectors.legsList.appendChild(card);
  });

  if (state.legs.length === 0) {
    selectors.modelProb.textContent = "--";
    selectors.impliedProb.textContent = "--";
    selectors.fairOdds.textContent = "--";
    selectors.expectedValue.textContent = "--";
    selectors.sampleSize.textContent = "--";
    return;
  }

  const jointModel = state.legs.reduce((acc, leg) => acc * leg.model_probability, 1);
  const implied = state.legs.reduce((acc, leg) => acc * americanToProb(leg.odds), 1);
  const fairOdds = probToAmerican(jointModel);
  const impliedOdds = probToAmerican(implied);
  const ev = expectedValue(jointModel, impliedOdds);

  selectors.modelProb.textContent = formatPercent(jointModel);
  selectors.impliedProb.textContent = formatPercent(implied);
  selectors.fairOdds.textContent = formatOdds(fairOdds);
  selectors.expectedValue.textContent = ev === null ? "--" : `${(ev * 100).toFixed(1)}%`;

  const minSample = Math.min(...state.legs.map((leg) => leg.sample_size || 0));
  selectors.sampleSize.textContent = minSample ? `${minSample} games` : "--";
};

init();
