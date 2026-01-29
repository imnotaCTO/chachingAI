const API_BASE = window.API_BASE || "http://127.0.0.1:8000";
const DEFAULT_STATS_SOURCE = window.DEFAULT_STATS_SOURCE || "kaggle";

const state = {
  data: null,
  eventId: null,
  sportsbook: "DraftKings",
  legs: [],
  loading: false,
  loadingProps: false,
  statsSource: DEFAULT_STATS_SOURCE,
  propsRequestId: 0,
  propsAbort: null,
  propsError: null,
  slateLoaded: false,
  propsView: "all",
  modelSettings: {
    window: 25,
    minGames: 15,
    minMinutes: 20,
    useEma: true,
    emaSpan: 12,
  },
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
  loadSlate: document.getElementById("loadSlate"),
  gamesList: document.getElementById("gamesList"),
  eventTitle: document.getElementById("eventTitle"),
  propsHint: document.getElementById("propsHint"),
  tabAll: document.getElementById("tabAll"),
  tabNotable: document.getElementById("tabNotable"),
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
  windowInput: document.getElementById("windowInput"),
  minGamesInput: document.getElementById("minGamesInput"),
  minMinutesInput: document.getElementById("minMinutesInput"),
  useEmaInput: document.getElementById("useEmaInput"),
  emaSpanInput: document.getElementById("emaSpanInput"),
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
  if (selectors.windowInput) {
    selectors.windowInput.value = state.modelSettings.window;
  }
  if (selectors.minGamesInput) {
    selectors.minGamesInput.value = state.modelSettings.minGames;
  }
  if (selectors.minMinutesInput) {
    selectors.minMinutesInput.value = state.modelSettings.minMinutes;
  }
  if (selectors.useEmaInput) {
    selectors.useEmaInput.checked = state.modelSettings.useEma;
  }
  if (selectors.emaSpanInput) {
    selectors.emaSpanInput.value = state.modelSettings.emaSpan;
    selectors.emaSpanInput.disabled = !state.modelSettings.useEma;
  }
  bindEvents();
  await loadSportsbooks();
  renderParlay();
  renderProps();
};

const bindEvents = () => {
  selectors.sportsbookSelect.addEventListener("change", (event) => {
    state.sportsbook = event.target.value;
    state.legs = [];
    if (state.slateLoaded && state.eventId) {
      loadProps(state.eventId);
    }
    renderParlay();
  });

  selectors.statsSource.addEventListener("change", (event) => {
    state.statsSource = event.target.value;
    state.legs = [];
    if (state.slateLoaded && state.eventId) {
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
      loadSportsbooks();
      if (state.slateLoaded) {
        loadEvents(date);
      } else {
        renderProps();
      }
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

  if (selectors.windowInput) {
    selectors.windowInput.addEventListener("change", (event) => {
      const value = Number(event.target.value);
      state.modelSettings.window = Number.isNaN(value) ? 0 : value;
      if (state.slateLoaded && state.eventId) {
        loadProps(state.eventId);
      }
    });
  }
  if (selectors.minGamesInput) {
    selectors.minGamesInput.addEventListener("change", (event) => {
      const value = Number(event.target.value);
      state.modelSettings.minGames = Number.isNaN(value) ? 1 : value;
      if (state.slateLoaded && state.eventId) {
        loadProps(state.eventId);
      }
    });
  }
  if (selectors.minMinutesInput) {
    selectors.minMinutesInput.addEventListener("change", (event) => {
      const value = Number(event.target.value);
      state.modelSettings.minMinutes = Number.isNaN(value) ? 0 : value;
      if (state.slateLoaded && state.eventId) {
        loadProps(state.eventId);
      }
    });
  }
  if (selectors.useEmaInput) {
    selectors.useEmaInput.addEventListener("change", (event) => {
      state.modelSettings.useEma = event.target.checked;
      if (selectors.emaSpanInput) {
        selectors.emaSpanInput.disabled = !state.modelSettings.useEma;
      }
      if (state.slateLoaded && state.eventId) {
        loadProps(state.eventId);
      }
    });
  }
  if (selectors.emaSpanInput) {
    selectors.emaSpanInput.addEventListener("change", (event) => {
      const value = Number(event.target.value);
      state.modelSettings.emaSpan = Number.isNaN(value) ? 10 : value;
      if (state.slateLoaded && state.eventId) {
        loadProps(state.eventId);
      }
    });
  }

  if (selectors.loadSlate) {
    selectors.loadSlate.addEventListener("click", async () => {
      const date = selectors.dateInput.value;
      state.slateLoaded = true;
      await loadEvents(date);
    });
  }
  if (selectors.tabAll) {
    selectors.tabAll.addEventListener("click", () => {
      state.propsView = "all";
      selectors.tabAll.classList.add("active");
      selectors.tabNotable?.classList.remove("active");
      renderProps();
    });
  }
  if (selectors.tabNotable) {
    selectors.tabNotable.addEventListener("click", () => {
      state.propsView = "notable";
      selectors.tabNotable.classList.add("active");
      selectors.tabAll?.classList.remove("active");
      renderProps();
    });
  }
};

const fetchJson = async (path, { signal } = {}) => {
  const response = await fetch(`${API_BASE}${path}`, { signal });
  if (!response.ok) {
    let message = `Request failed: ${response.status}`;
    try {
      const payload = await response.json();
      if (payload?.error) {
        message = payload.error;
      }
    } catch (error) {
      // ignore parse errors
    }
    throw new Error(message);
  }
  return response.json();
};

const loadSportsbooks = async () => {
  try {
    const dateValue = selectors.dateInput?.value;
    const dateParam = dateValue ? `?date=${encodeURIComponent(dateValue)}` : "";
    const data = await fetchJson(`/api/sportsbooks${dateParam}`);
    state.data = state.data || {};
    state.data.sportsbooks = data.sportsbooks || [];
    if (!state.sportsbook) {
      state.sportsbook = "ALL";
    } else if (
      state.sportsbook !== "ALL" &&
      state.data.sportsbooks.length > 0 &&
      !state.data.sportsbooks.includes(state.sportsbook)
    ) {
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
  const timeout = window.setTimeout(() => controller.abort(), 20000);
  try {
    const sportsbookParam =
      state.sportsbook && state.sportsbook !== "ALL"
        ? `sportsbook=${encodeURIComponent(state.sportsbook)}&`
        : "";
    const statsSource = encodeURIComponent(state.statsSource || "kaggle");
    const settings = state.modelSettings;
    const settingsParams = new URLSearchParams({
      stats_source: statsSource,
      max_players: "20",
      window: String(settings.window ?? 0),
      min_games: String(settings.minGames ?? 1),
      min_minutes: String(settings.minMinutes ?? 0),
      use_ema: settings.useEma ? "1" : "0",
      ema_span: String(settings.emaSpan ?? 10),
    }).toString();
    const data = await fetchJson(
      `/api/events/${eventId}/props?${sportsbookParam}${settingsParams}`,
      { signal: controller.signal }
    );
    if (requestId !== state.propsRequestId) {
      return;
    }
    state.data = state.data || {};
    state.data.props = data.props || [];
    state.data.propsMeta = {
      availableMarkets: data.available_markets || [],
      bookmakers: data.bookmakers || [],
      warning: data.warning || null,
    };
    selectors.lastUpdate.textContent = data.last_update || "--";
    const availableBooks = Array.from(
      new Set(state.data.props.map((prop) => prop.sportsbook).filter(Boolean))
    ).sort();
    if (availableBooks.length) {
      state.data.sportsbooks = availableBooks;
      if (state.sportsbook !== "ALL" && !availableBooks.includes(state.sportsbook)) {
        state.sportsbook = "ALL";
        renderSportsbooks();
        showToast("Selected sportsbook unavailable for this event. Showing all.", "error");
      } else {
        renderSportsbooks();
      }
    }
    renderProps();
  } catch (error) {
    if (requestId !== state.propsRequestId) {
      return;
    }
    let message = error?.message || "Unable to load props.";
    if (error?.name === "AbortError") {
      message = "Props request timed out. Try Kaggle or reduce max players.";
    }
    state.propsError = message;
    showToast(message, "error");
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
  if (!state.slateLoaded) {
    selectors.eventTitle.textContent = "Load the slate";
    selectors.propsList.innerHTML = `<div class="subtle">Pick sportsbook + stats source, then click “Load slate”.</div>`;
    return;
  }
  selectors.eventTitle.textContent = event
    ? `${event.away_team} @ ${event.home_team}`
    : "Select a game";
  if (selectors.propsHint) {
    selectors.propsHint.textContent =
      state.propsView === "notable"
        ? "Notable bets: +EV with line on the right side of the predicted mean."
        : "Choose legs for one sportsbook";
  }
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

  const isNotable = (prop) => {
    if (prop.model_probability === null || prop.model_probability === undefined) return false;
    if (prop.model_mean === null || prop.model_mean === undefined) return false;
    if (prop.line === null || prop.line === undefined) return false;
    if (prop.odds === null || prop.odds === undefined) return false;
    const evValue = expectedValue(prop.model_probability, prop.odds);
    if (evValue === null || evValue <= 0) return false;
    const direction = String(prop.direction || "over").toLowerCase();
    if (direction === "over") {
      return Number(prop.line) < Number(prop.model_mean);
    }
    if (direction === "under") {
      return Number(prop.line) > Number(prop.model_mean);
    }
    return false;
  };

  const propsForView =
    state.propsView === "notable" ? filteredProps.filter(isNotable) : filteredProps;

  const grouped = new Map();
  propsForView.forEach((prop) => {
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
      const meta = state.data?.propsMeta;
      const markets =
        meta?.availableMarkets && meta.availableMarkets.length
          ? `Available markets: ${meta.availableMarkets.join(", ")}.`
          : "";
      const books =
        meta?.bookmakers && meta.bookmakers.length
          ? `Bookmakers: ${meta.bookmakers.join(", ")}.`
          : "";
      const warning = meta?.warning ? `${meta.warning} ` : "";
      selectors.propsList.innerHTML = `<div class="subtle">No props returned from API for this event. ${warning}${markets} ${books}</div>`;
    } else if (state.propsView === "notable") {
      selectors.propsList.innerHTML = `<div class="subtle">No notable bets found for this event.</div>`;
    } else if (hasFilters) {
      selectors.propsList.innerHTML = `<div class="subtle">No props match the filters.</div>`;
    } else {
      selectors.propsList.innerHTML = `<div class="subtle">No props available for this event.</div>`;
    }
    return;
  }

  const statOrder = { points: 0, rebounds: 1, assists: 2 };
  const groups = Array.from(grouped.values()).map((group) => {
    const lineDirections = new Map();
    group.items.forEach((prop) => {
      const lineKey = String(prop.line);
      const direction = String(prop.direction || "over").toLowerCase();
      if (!lineDirections.has(lineKey)) {
        lineDirections.set(lineKey, new Set());
      }
      lineDirections.get(lineKey).add(direction);
    });
    const ouItems = [];
    const xPlusItems = [];
    group.items.forEach((prop) => {
      const lineKey = String(prop.line);
      const directions = lineDirections.get(lineKey) || new Set();
      const direction = String(prop.direction || "over").toLowerCase();
      if (direction === "over" && !directions.has("under")) {
        xPlusItems.push(prop);
      } else {
        ouItems.push(prop);
      }
    });
    const firstWithAvg = group.items.find((item) => item.season_avg !== null && item.season_avg !== undefined);
    const firstWithModel = group.items.find((item) => item.model_mean !== null && item.model_mean !== undefined);
    return {
      player_name: group.player_name,
      stat: group.stat,
      ouItems,
      xPlusItems,
      season_avg: firstWithAvg ? firstWithAvg.season_avg : null,
      model_mean: firstWithModel ? firstWithModel.model_mean : null,
    };
  });

  const renderSection = (title, groupsToRender, addPlus) => {
    const section = document.createElement("div");
    section.className = "props-section";
    section.innerHTML = `<div class="section-title">${title}</div>`;
    if (!groupsToRender.length) {
      const note = document.createElement("div");
      note.className = "section-note";
      note.textContent = "No lines available.";
      section.appendChild(note);
      selectors.propsList.appendChild(section);
      return;
    }
    groupsToRender.forEach((group) => {
      const row = document.createElement("div");
      row.className = "prop-row";
      const seasonAvg =
        group.season_avg === null || group.season_avg === undefined
          ? "--"
          : Number(group.season_avg).toFixed(1);
      const modelMean =
        group.model_mean === null || group.model_mean === undefined
          ? "--"
          : Number(group.model_mean).toFixed(1);
      row.innerHTML = `
        <div>
          <div class="player">${group.player_name}</div>
          <div class="meta">${group.stat.toUpperCase()}</div>
          <div class="meta">Avg: ${seasonAvg} · Predicted: ${modelMean}</div>
        </div>
        <div class="prop-options"></div>
      `;
      const options = row.querySelector(".prop-options");
      const items = [...(addPlus ? group.xPlusItems : group.ouItems)].sort(
        (a, b) => a.line - b.line
      );
      items.forEach((prop) => {
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
        const plusTag = addPlus ? "+" : "";
        option.innerHTML = `
          <div class="line">${prop.direction} ${prop.line}${plusTag}</div>
          <div class="odds">${formatOdds(prop.odds)}</div>
          <div class="meta">${prop.sportsbook || "Sportsbook"}</div>
          <div class="ev">${evValue === null ? "--" : `${(evValue * 100).toFixed(1)}% EV`}</div>
        `;
        option.addEventListener("click", () => addLeg(prop));
        options.appendChild(option);
      });
      section.appendChild(row);
    });
    selectors.propsList.appendChild(section);
  };

  const sortGroups = (a, b) => {
    const statA = statOrder[a.stat] ?? 99;
    const statB = statOrder[b.stat] ?? 99;
    if (statA !== statB) return statA - statB;
    return a.player_name.localeCompare(b.player_name);
  };
  const ouGroups = groups.filter((group) => group.ouItems.length).sort(sortGroups);
  const xPlusGroups = groups.filter((group) => group.xPlusItems.length).sort(sortGroups);

  renderSection("X+ lines", xPlusGroups, true);
  renderSection("O/U lines", ouGroups, false);
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
